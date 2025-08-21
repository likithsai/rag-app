FROM node:18

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y build-essential python3 git

COPY ./package*.json ./

# Force build hnswlib-node from source (ignore prebuilds)
RUN npm install --build-from-source=hnswlib-node --legacy-peer-deps --verbose

COPY . .

COPY .env .env

EXPOSE ${PORT:-5001}

CMD ["npx", "ts-node", "src/server.ts"]