- When you first git pull this repo:
pip install -r (path to requirements.txt)
- If video processing was updated, run the same command again so H.264/ffmpeg support is installed.
- Download the cloudfare tunnel:
https://developers.cloudflare.com/tunnel/downloads/
- Rename the file to cloudflared.exe
- Add it to path environment variables
- Step 1: Run server:
uvicorn main:app --host 0.0.0.0 --port 8000
- Step 2: Run cloud flare:
cloudflared tunnel --url http://localhost:8000
- Step 3: Copy the url cloudflared gave you and paste it to local properties in android app 
opticalFlowServerBaseUrl=https://testimony-bufing-photos-carry.trycloudflare.com (example)
- Step 4: Run the app and done.


