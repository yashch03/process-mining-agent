# BrowserGym/MiniWoB++ environment setup

If MINIWOB_URL errors with "file not found," the MiniWoB HTML files need
re-cloning — Lightning Studios do NOT persist /tmp/ across sessions.

Fix:
cd ~
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
export MINIWOB_URL="file://$HOME/miniwob-plusplus/miniwob/html/miniwob/"
Also requires (one-time, persists across sessions):
sudo apt-get install -y libnss3 libnspr4 libasound2t64
