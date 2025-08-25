FROM node:18

WORKDIR /app

# Install build tools required for hnswlib-node
RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

# Copy only package.json first for caching
COPY package*.json ./

# Install deps (force build hnswlib-node from source)
RUN npm install --build-from-source=hnswlib-node --legacy-peer-deps --verbose

# Copy rest of app
COPY . .

# Build TypeScript into dist/
RUN npm run build

EXPOSE ${PORT:-5001}

# Run compiled JS, not ts-node
CMD ["node", "dist/server.js"]