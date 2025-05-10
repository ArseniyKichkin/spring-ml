#!/bin/bash

# URL вашего API
API_URL="http://localhost:5000/predict"


generate_correct_data() {
  local salary=$((RANDOM % 50 + 50))
  local created_at=$(date +%s%6N)

  cat <<EOF
{
  "created_at": $created_at,
  "salary": $salary
}
EOF
}


while true; do

  data=$(generate_correct_data)


  response=$(curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "$data")


  timestamp=$(date "+%Y-%m-%d %H:%M:%S")
  echo "[$timestamp] Sent data: $data"
  echo "[$timestamp] Response: $response"


  sleep 1
done
