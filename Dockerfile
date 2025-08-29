FROM node:18

WORKDIR /app

# Copy only package.json first for caching
COPY package*.json ./

# Install deps (force build hnswlib-node from source)
RUN npm install --legacy-peer-deps --verbose

# Copy rest of app
COPY . .

# Build TypeScript into dist/
RUN npm run build

EXPOSE ${PORT:-5001}

# Run compiled JS, not ts-node
CMD ["node", "dist/server.min.js"]