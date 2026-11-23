import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const testsText = await fs.readFile(path.join(root, '테스트질문.md'), 'utf8');
const index = JSON.parse(await fs.readFile(path.join(root, 'data', 'index.json'), 'utf8'));
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const tests = [...testsText.matchAll(/^\|\s*(\d+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|$/gm)]
  .map((m) => ({ number: Number(m[1]), type: m[2].trim(), question: m[3].trim(), expected: m[4].trim() }));

function cosine(a, b) {
  let dot = 0; let aa = 0; let bb = 0;
  for (let i = 0; i < a.length; i += 1) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return dot / (Math.sqrt(aa) * Math.sqrt(bb));
}

const embeddingResponse = await client.embeddings.create({
  model: 'text-embedding-3-small',
  input: tests.map((test) => test.question),
  encoding_format: 'float'
});

const results = [];
for (let i = 0; i < tests.length; i += 1) {
  const test = tests[i];
  const ranked = index.records
    .map((record) => ({ id: record.id, score: cosine(embeddingResponse.data[i].embedding, record.embedding) }))
    .sort((a, b) => b.score - a.score);
  const expectedId = test.expected === '없음' ? null : test.expected.replace(/제(\d+)조/g, '$1').replaceAll(' ', '-');
  const rank = expectedId ? ranked.findIndex((item) => item.id === expectedId) + 1 : null;
  const response = await fetch('http://localhost:3000/api/chat', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ question: test.question })
  });
  const answer = await response.json();
  results.push({
    number: test.number, type: test.type, question: test.question, expected: expectedId || '없음',
    rank: rank || '-', topScore: Number(ranked[0].score.toFixed(4)),
    topResult: ranked[0].id, withinTop5: expectedId ? rank <= 5 : null,
    answerNotFound: answer.answer === '규정에서 찾지 못했습니다'
  });
}

console.log(JSON.stringify(results));
