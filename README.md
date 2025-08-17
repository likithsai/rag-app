RAG App (PDF-based QA using Ollama + LangChain)

A Retrieval-Augmented Generation (RAG) backend that loads PDFs from a folder, embeds their content, and answers questions using a local Ollama LLM. Built with TypeScript, Express, and LangChain.

⸻

Features
• Recursively scans a folder for PDFs (including subfolders).
• Processes PDFs in batches for better performance.
• Splits PDF content into chunks and generates embeddings using OllamaEmbeddings.
• Answers questions based on similarity search of embedded PDF content.
• Fully configurable via .env.
• TypeScript + ts-node development workflow.
• Optional: auto-reload during development with nodemon.

⸻

Folder Structure

rag-app/
├─ backend/
│ ├─ server.ts # Main backend server
│ └─ config.ts # Env-based configuration
├─ dist/ # Compiled JavaScript output
├─ pdfs/ # Place all PDFs here (subfolders supported)
├─ package.json
├─ tsconfig.json
└─ .env

⸻

Prerequisites
• Node.js >= 20
• npm >= 9
• Ollama installed locally: https://ollama.com/

⸻

Installation 1. Clone the repository:

git clone <repo-url>
cd rag-app

    2.	Install dependencies:

npm install

    3.	Create .env file in the root:

PORT=3001
PUBLIC_FOLDER=./backend/pdfs
OLLAMA_MODEL=llama3.1
EMBEDDING_MODEL=nomic-embed-text

⸻

Development

Run the server in dev mode with automatic reload:

npm run dev

    •	Uses ts-node + nodemon.
    •	Watches changes in backend/ folder.

⸻

Build & Production 1. Build TypeScript into JavaScript:

npm run build

    •	Compiles TS files into dist/ folder.

    2.	Start the server:

npm run start

    •	Runs dist/server.js.
    •	Ensure your PDFs folder is populated before starting.

⸻

API

POST /ask

Ask a question based on the loaded PDFs.
• Request body:

{
"question": "What is the main topic of file1.pdf?"
}

    •	Response:

{
"answer": "The main topic of file1.pdf is ...",
"sources": ["file1.pdf", "file2.pdf"]
}

⸻

Notes
• Supports large numbers of PDFs using batch concurrency to limit memory usage.
• Keeps source metadata for each text chunk.
• Recursive PDF scanning allows organization in subfolders.
• Uses LangChain MemoryVectorStore for similarity search.

⸻

Optional Improvements
• Add file watcher to automatically embed new PDFs without restarting.
• Persist vector store to disk for faster startup.
• Add frontend UI for asking questions.

⸻

License

MIT License

⸻

I can also create a version with a minimal UI example that queries the backend directly, so you can ask questions from a browser.

Do you want me to do that?

curl -X POST http://localhost:3001/ask \
 -H "Content-Type: application/json" \
 -d '{
"question": "What is the summary of document X?"
}'
