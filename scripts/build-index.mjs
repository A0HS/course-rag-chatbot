import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const regulationsDir = path.join(root, '규정');
const outputPath = path.join(root, 'data', 'index.json');

if (!process.env.OPENAI_API_KEY) {
  throw new Error('OPENAI_API_KEY 환경변수가 필요합니다.');
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const files = (await fs.readdir(regulationsDir))
  .filter((file) => file.endsWith('.md'))
  .sort();

function parseArticles(markdown, regulationName) {
  const lines = markdown.split(/\r?\n/);
  const articles = [];
  let chapter = '';
  let current = null;

  const save = () => {
    if (!current) return;
    const body = current.body.join('\n').trim();
    const tableLines = current.body.filter((line) => /^\s*\|.*\|\s*$/.test(line));
    const searchableText = [
      regulationName,
      chapter,
      current.title,
      body.replace(/\|/g, ' ')
    ].filter(Boolean).join('\n');
    articles.push({
      id: `${regulationName}-${current.number}`,
      regulation: regulationName,
      chapter,
      article: `제${current.number}조`,
      title: current.title,
      text: body,
      hasTable: tableLines.length >= 2,
      searchableText
    });
  };

  for (const line of lines) {
    const chapterMatch = line.match(/^##\s+(제\d+장\s+.+)$/);
    const articleMatch = line.match(/^###\s+(제(\d+)조(?:\([^)]*\))?)/);
    if (chapterMatch) {
      save();
      current = null;
      chapter = chapterMatch[1].trim();
      continue;
    }
    if (articleMatch) {
      save();
      current = {
        number: articleMatch[2],
        title: articleMatch[1].trim(),
        body: []
      };
      continue;
    }
    if (current) current.body.push(line);
  }
  save();
  return articles;
}

const records = [];
for (const file of files) {
  const markdown = await fs.readFile(path.join(regulationsDir, file), 'utf8');
  const regulationName = path.basename(file, '.md');
  records.push(...parseArticles(markdown, regulationName));
}

const embeddings = [];
for (let start = 0; start < records.length; start += 100) {
  const batch = records.slice(start, start + 100);
  const response = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: batch.map((record) => record.searchableText),
    encoding_format: 'float'
  });
  embeddings.push(...response.data.sort((a, b) => a.index - b.index).map((item) => item.embedding));
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify({
  version: 1,
  model: 'text-embedding-3-small',
  createdAt: new Date().toISOString(),
  records: records.map((record, index) => ({ ...record, embedding: embeddings[index] }))
}, null, 2), 'utf8');

console.log(`인덱스 생성 완료: ${records.length}개 조항 -> ${outputPath}`);
