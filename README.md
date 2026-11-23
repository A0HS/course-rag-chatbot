# 규정 챗봇 v1

## 실행

PowerShell에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```powershell
npm install
npm run build:index
npm start
```

브라우저에서 http://localhost:3001 을 엽니다.

`OPENAI_API_KEY`는 PC 환경변수로 설정하거나 `.env` 파일에 넣습니다. `.env`는 Git에 올리지 않습니다.

## Vercel 배포

이 프로젝트는 `api/index.mjs`가 Express 앱을 Vercel 함수로 내보내는 구조입니다. GitHub 저장소를 Vercel에 연결한 뒤 별도 빌드 명령 없이 배포할 수 있습니다.

Vercel 프로젝트 환경변수에 다음 값을 등록합니다.

```text
OPENAI_API_KEY=실제_OpenAI_API_키
OPENAI_CHAT_MODEL=gpt-4o-mini
MIN_SIMILARITY=0.25
RATE_LIMIT_MAX=10
RATE_LIMIT_WINDOW_MS=60000
```

`RATE_LIMIT_MAX`와 `RATE_LIMIT_WINDOW_MS`는 IP별 요청 제한입니다. 기본값은 1분당 10회입니다. 현재 구현은 서버 인스턴스 메모리 기반이므로 여러 Vercel 인스턴스에 걸친 전역 제한이 필요하면 Redis/Upstash 같은 공유 저장소를 추가해야 합니다.

## 동작

- `규정/*.md`를 제1조, 제2조처럼 조항 단위로 분할합니다.
- 조항별 규정명, 장, 조 제목, 표 포함 여부를 보존합니다.
- `text-embedding-3-small`로 `data/index.json`을 생성합니다.
- 질문도 같은 모델로 임베딩하고, 한국어 bigram 기반 BM25 키워드 검색을 함께 수행합니다.
- 벡터 후보 10개와 BM25 후보 3개를 RRF(k=60)로 합칩니다. 벡터 순위에 가중치를 주고, BM25 고득점 결과 1개는 앵커로 보존해 의미 검색이 밀리지 않게 최종 상위 5개를 고릅니다.
- 검색 결과를 근거로 답변하고, 답변 아래에 근거 조항을 표시합니다.
- 전체 조항 중 최고 벡터 유사도가 기준보다 낮으면 BM25 결과와 관계없이 `규정에서 찾지 못했습니다`를 반환합니다.

## Vercel 배포를 고려한 구조

- API 키를 브라우저 코드에 넣지 않고 서버 라우트에서만 사용합니다.
- `/api/chat`은 서버리스 함수로 옮기기 쉬운 HTTP API 형태입니다.
- 현재는 로컬의 `data/index.json`을 읽습니다. 배포 시에는 인덱스를 저장소에 포함하거나 Blob/DB로 옮기면 됩니다.
