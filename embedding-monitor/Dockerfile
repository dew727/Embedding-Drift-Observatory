FROM node:20-alpine

WORKDIR /app

COPY . .
RUN npm ci
RUN npm run build

RUN npm install -g serve

CMD ["sh", "-c", "serve -s dist -l ${PORT:-3000}"]
