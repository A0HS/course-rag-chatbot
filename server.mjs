import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import express from 'express';
import OpenAI from 'openai';

dotenv.config();
const root = path.dirname(fileURLToPath(import.meta.url));
const indexPath = path.join(root, 'data', 'index.json');
const port = Number(process.env.PORT || 3001);
const minSimilarity = Number(process.env.MIN_SIMILARITY || 0.25);
const rrfK = 60;
const retrievalLimit = 5;
const rrfCandidateLimit = Number(process.env.RRF_CANDIDATE_LIMIT || 10);
const keywordCandidateLimit = Number(process.env.RRF_KEYWORD_CANDIDATE_LIMIT || 3);
const vectorWeight = Number(process.env.RRF_VECTOR_WEIGHT || 1.5);
const keywordWeight = Number(process.env.RRF_KEYWORD_WEIGHT || 1);
const keywordAnchorScore = Number(process.env.RRF_KEYWORD_ANCHOR_SCORE || 8);
const rateLimitWindowMs = Number(process.env.RATE_LIMIT_WINDOW_MS || 60_000);
const rateLimitMax = Number(process.env.RATE_LIMIT_MAX || 10);
const serverApiKey = process.env.OPENAI_API_KEY?.trim() || '';
const app = express();
const requestCounts = new Map();

app.set('trust proxy', true);
app.use(express.json({ limit: '32kb' }));
app.use(express.static(path.join(root, 'public')));

app.get('/api/config', (_req, res) => {
  // Only expose whether a server key exists; never send the key to the browser.
  res.json({ serverApiKeyConfigured: Boolean(serverApiKey) });
});

app.use('/api/chat', (req, res, next) => {
  const forwardedFor = req.headers['x-forwarded-for'];
  const clientIp = (typeof forwardedFor === 'string' ? forwardedFor.split(',')[0].trim() : '')
    || req.ip || req.socket.remoteAddress || 'unknown';
  const now = Date.now();
  const current = requestCounts.get(clientIp);
  const windowStart = current && now - current.windowStart < rateLimitWindowMs ? current.windowStart : now;
  const count = current && windowStart === current.windowStart ? current.count + 1 : 1;
  requestCounts.set(clientIp, { windowStart, count });
  if (count > rateLimitMax) {
    const retryAfter = Math.max(1, Math.ceil((windowStart + rateLimitWindowMs - now) / 1000));
    res.set('Retry-After', String(retryAfter));
    return res.status(429).json({ error: `질문은 1분에 ${rateLimitMax}번까지 가능합니다.` });
  }
  return next();
});

function cosineSimilarity(a, b) {
  let dot = 0; let normA = 0; let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Index Korean text as overlapping character bigrams; use word tokens for other text.
function tokenize(text) {
  const normalized = String(text || '').normalize('NFKC').toLowerCase();
  const tokens = [];
  for (const match of normalized.matchAll(/[가-힣]+|[a-z0-9]+/g)) {
    const token = match[0];
    if (/^[가-힣]+$/.test(token)) {
      if (token.length === 1) tokens.push(token);
      for (let i = 0; i < token.length - 1; i += 1) tokens.push(token.slice(i, i + 2));
    } else {
      tokens.push(token);
    }
  }
  return tokens;
}

function createBm25(records) {
  const documents = records.map((record) => tokenize(record.searchableText));
  const documentFrequency = new Map();
  const averageLength = documents.reduce((sum, tokens) => sum + tokens.length, 0) / Math.max(documents.length, 1);
  for (const tokens of documents) {
    for (const token of new Set(tokens)) documentFrequency.set(token, (documentFrequency.get(token) || 0) + 1);
  }

  return (question) => {
    const query = tokenize(question);
    const queryTokens = [...new Set(query)];
    const queryFrequency = new Map();
    for (const token of query) queryFrequency.set(token, (queryFrequency.get(token) || 0) + 1);

    return records.map((record, index) => {
      const termFrequency = new Map();
      for (const token of documents[index]) termFrequency.set(token, (termFrequency.get(token) || 0) + 1);
      let score = 0;
      for (const token of queryTokens) {
        const df = documentFrequency.get(token) || 0;
        const tf = termFrequency.get(token) || 0;
        if (!df || !tf) continue;
        const idf = Math.log(1 + (records.length - df + 0.5) / (df + 0.5));
        const boostedTf = tf * (queryFrequency.get(token) || 1);
        const denominator = boostedTf + 1.2 * (1 - 0.75 + 0.75 * documents[index].length / Math.max(averageLength, 1));
        score += idf * (boostedTf * 2.2) / denominator;
      }
      return { ...record, bm25Score: score };
    }).sort((a, b) => b.bm25Score - a.bm25Score).slice(0, keywordCandidateLimit);
  };
}

function hybridSearch(records, question, queryEmbedding) {
  const vectorResults = records
    .map((record) => ({ ...record, vectorScore: cosineSimilarity(queryEmbedding, record.embedding) }))
    .sort((a, b) => b.vectorScore - a.vectorScore).slice(0, rrfCandidateLimit);
  const keywordResults = createBm25(records)(question);
  const ranked = new Map();
  for (const [rank, record] of vectorResults.entries()) {
    ranked.set(record.id, { ...record, rrfScore: vectorWeight / (rrfK + rank + 1) });
  }
  for (const [rank, record] of keywordResults.entries()) {
    const current = ranked.get(record.id) || { ...record, rrfScore: 0 };
    ranked.set(record.id, { ...current, bm25Score: record.bm25Score, rrfScore: current.rrfScore + keywordWeight / (rrfK + rank + 1) });
  }
  const fusedResults = [...ranked.values()].sort((a, b) => b.rrfScore - a.rrfScore);
  const anchors = keywordResults.filter((record) => record.bm25Score >= keywordAnchorScore).slice(0, 1);
  const selected = [];
  for (const record of [...anchors, ...fusedResults]) {
    if (!selected.some((item) => item.id === record.id)) selected.push(record);
    if (selected.length === retrievalLimit) break;
  }
  return selected;
}

function formatContext(results) {
  return results.map((item) =>
    `[${item.regulation} · ${item.article} ${item.title}]\n${item.text}`
  ).join('\n\n');
}

app.post('/api/chat', async (req, res) => {
  try {
    const question = String(req.body?.question || '').trim();
    if (!question) return res.status(400).json({ error: '질문을 입력해 주세요.' });

    // Prefer the server key. A browser key is used only for this request and is
    // never written to disk, logged, or included in a response.
    const requestApiKey = serverApiKey || String(req.body?.apiKey || '').trim();
    if (!requestApiKey) {
      return res.status(400).json({ error: 'OpenAI API 키를 입력하거나 서버 환경변수 OPENAI_API_KEY를 설정해 주세요.' });
    }
    const client = new OpenAI({ apiKey: requestApiKey });

    const index = JSON.parse(await fs.readFile(indexPath, 'utf8'));
    const embeddingResponse = await client.embeddings.create({
      model: 'text-embedding-3-small',
      input: question,
      encoding_format: 'float'
    });
    const queryEmbedding = embeddingResponse.data[0].embedding;
    const results = hybridSearch(index.records, question, queryEmbedding);
    const bestVectorScore = Math.max(...index.records.map((record) => cosineSimilarity(queryEmbedding, record.embedding)));

    if (!results.length || bestVectorScore < minSimilarity) {
      return res.json({ answer: '규정에서 찾지 못했습니다', sources: [] });
    }

    const response = await client.responses.create({
      model: process.env.OPENAI_CHAT_MODEL || 'gpt-4o-mini',
      instructions: [
        '너는 회사 규정 안내 챗봇이다.',
        '아래 근거 조항에 있는 내용만 사용해 한국어로 답한다.',
        '근거에 없는 내용은 추측하지 않는다.',
        '질문에 답할 수 없으면 정확히 "규정에서 찾지 못했습니다"라고 답한다.',
        '답변 본문에는 조항 출처를 반복하지 말고, 출처는 별도 영역에서 표시한다.'
      ].join('\n'),
      input: `질문:\n${question}\n\n근거 조항 5개:\n${formatContext(results)}`
    });

    const answer = response.output_text?.trim() || '규정에서 찾지 못했습니다';
    const notFound = answer.replace(/[\s.!。]+$/g, '') === '규정에서 찾지 못했습니다';
    res.json({
      answer: notFound ? '규정에서 찾지 못했습니다' : answer,
      sources: notFound ? [] : results.map(({ embedding, searchableText, ...source }) => source)
    });
  } catch (error) {
    // Log only the status and message. Never log the request body (it may contain a key).
    console.error('chat error', error?.status || '', error?.message || error);
    if (error?.status === 401) {
      return res.status(401).json({ error: 'OpenAI API 키가 올바르지 않습니다. 키를 다시 확인해 주세요.' });
    }
    res.status(500).json({ error: '답변을 만드는 중 오류가 발생했습니다.' });
  }
});

app.get('/api/health', async (_req, res) => {
  try {
    const index = JSON.parse(await fs.readFile(indexPath, 'utf8'));
    res.json({ ok: true, articles: index.records.length, embeddingModel: index.model });
  } catch {
    res.status(503).json({ ok: false, message: '먼저 npm run build:index를 실행해 주세요.' });
  }
});

export { app };
export default app;

if (process.env.VERCEL !== '1') {
  app.listen(port, () => {
    console.log(`규정 챗봇 실행 중: http://localhost:${port}`);
  });
}
