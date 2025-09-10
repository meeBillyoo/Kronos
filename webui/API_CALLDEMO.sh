#!/bin/bash

# 1. 加载模型
echo "Loading kronos-base model..."
curl -X POST http://localhost:7070/api/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kronos-base",
    "device": "cpu"
  }'

echo -e "\n\n"

# 2. 获取K线数据并保存到临时文件
echo "Fetching BTC-USDT 4H kline data..."
kline_response=$(curl -s -X POST http://localhost:7070/api/exchange-kline \
  -H "Content-Type: application/json" \
  -d '{
    "exchange": "binance",
    "symbol": "ETH-USDT",
    "timeframe": "1H",
    "limit": 500
  }')

echo "$kline_response"
echo -e "\n\n"

# 3. 提取kline_data并进行预测
echo "Making prediction with kline data..."
kline_data=$(echo "$kline_response" | jq '.data.kline_data')

curl -X POST http://localhost:7070/api/predictkline \
  -H "Content-Type: application/json" \
  -d "{
    \"kline_data\": $kline_data,
    \"lookback\": 400,
    \"pred_len\": 100,
    \"temperature\": 1.0,
    \"top_p\": 0.9,
    \"sample_count\": 1
  }"