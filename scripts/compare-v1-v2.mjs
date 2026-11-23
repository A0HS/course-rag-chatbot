import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const testsText = await fs.readFile(path.join(root, '테스트질문.md'), 'utf8');
const index = JSON.parse(await fs.readFile(path.join(root, 'data', 'index.json'), 'utf8'));
const tests = [...testsText.matchAll(/^\|\s*(\d+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|$/gm)]
  .map((m) => ({ number: Number(m[1]), type: m[2].trim(), question: m[3].trim(), expected: m[4].trim() }));
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function cosine(a, b) {
  let dot = 0; let aa = 0; let bb = 0;
  for (let i = 0; i < a.length; i += 1) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return dot / (Math.sqrt(aa) * Math.sqrt(bb));
}

function tokenize(text) {
  const tokens = [];
  for (const match of String(text || '').normalize('NFKC').toLowerCase().matchAll(/[가-힣]+|[a-z0-9]+/g)) {
    const token = match[0];
    if (/^[가-힣]+$/.test(token)) {
      if (token.length === 1) tokens.push(token);
      for (let i = 0; i < token.length - 1; i += 1) tokens.push(token.slice(i, i + 2));
    } else tokens.push(token);
  }
  return tokens;
}

function bm25(records, question) {
  const docs = records.map((record) => tokenize(record.searchableText));
  const df = new Map();
  for (const doc of docs) for (const token of new Set(doc)) df.set(token, (df.get(token) || 0) + 1);
  const avg = docs.reduce((sum, doc) => sum + doc.length, 0) / docs.length;
  const query = tokenize(question);
  return records.map((record, i) => {
    const tf = new Map();
    for (const token of docs[i]) tf.set(token, (tf.get(token) || 0) + 1);
    let score = 0;
    for (const token of new Set(query)) {
      const frequency = tf.get(token) || 0; const documentFrequency = df.get(token) || 0;
      if (!frequency || !documentFrequency) continue;
      const idf = Math.log(1 + (records.length - documentFrequency + 0.5) / (documentFrequency + 0.5));
      score += idf * (frequency * 2.2) / (frequency + 1.2 * (1 - 0.75 + 0.75 * docs[i].length / avg));
    }
    return { id: record.id, score };
  }).sort((a, b) => b.score - a.score).slice(0, 5);
}

const embeddings = await client.embeddings.create({
  model: 'text-embedding-3-small', input: tests.map((test) => test.question), encoding_format: 'float'
});

const rows = tests.map((test, i) => {
  const vector = index.records.map((record) => ({ id: record.id, score: cosine(embeddings.data[i].embedding, record.embedding) }))
    .sort((a, b) => b.score - a.score);
  const keyword = bm25(index.records, test.question);
  const ranked = new Map();
  vector.slice(0, 5).forEach((item, rank) => ranked.set(item.id, { id: item.id, rrf: 1 / (60 + rank + 1) }));
  keyword.forEach((item, rank) => ranked.set(item.id, { ...(ranked.get(item.id) || { id: item.id, rrf: 0 }), rrf: (ranked.get(item.id)?.rrf || 0) + 1 / (60 + rank + 1) }));
  const hybrid = [...ranked.values()].sort((a, b) => b.rrf - a.rrf);
  const expected = test.expected === '없음' ? null : test.expected.replace(/제(\d+)조/g, '$1').replaceAll(' ', '-');
  const v1Rank = expected ? vector.findIndex((item) => item.id === expected) + 1 : null;
  const v2Rank = expected ? hybrid.findIndex((item) => item.id === expected) + 1 : null;
  const v1Correct = expected ? v1Rank > 0 && v1Rank <= 5 : vector[0].score < 0.25;
  const v2Correct = expected ? v2Rank > 0 && v2Rank <= 5 : hybrid.every((item) => !keyword.some((k) => k.id === item.id && k.score > 0));
  return {
    number: test.number, type: test.type, question: test.question, expected: test.expected,
    v1Rank: v1Rank || '-', v2Rank: v2Rank || '-', v1Top: vector[0].id, v2Top: hybrid[0].id,
    v1Correct, v2Correct, newlyFound: !v1Correct && v2Correct
  };
});

const apiChecks = await Promise.all(rows.map(async (row) => {
  const response = await fetch('http://localhost:3001/api/chat', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ question: row.question })
  });
  const body = await response.json();
  return { number: row.number, answerNotFound: body.answer === '규정에서 찾지 못했습니다', sourceCount: body.sources?.length || 0 };
}));
for (const check of apiChecks) Object.assign(rows.find((row) => row.number === check.number), check);

console.log(JSON.stringify(rows));
