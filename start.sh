xhost +local:root
docker compose up -d --build && docker compose exec app /bin/bash