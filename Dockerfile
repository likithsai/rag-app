# Use Node 20 slim for smaller image
FROM node:18

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential python3
RUN npm install hnswlib-node

COPY ./package*.json ./

# Install dependencies ignoring peer dependency conflicts
RUN npm ci --legacy-peer-deps --vebrose

# Copy source code
COPY . .

# Copy .env file into container
COPY .env .env

# Expose port from .env (default fallback 5000)
EXPOSE ${PORT:-5001}

# Run the server
CMD ["npx", "ts-node", "src/server.ts"]