# Dashboard viewing setup

Lightning AI's built-in port-forwarding (Ports panel -> Forward a Port) does
NOT reliably serve Streamlit's dashboard through the browser — the server
starts and reports healthy, but the page renders blank. This is a known
class of issue: Streamlit relies on WebSockets and root-path assumptions
that Lightning's proxy layer does not handle correctly.

## Working fix: use ngrok to bypass Lightning's proxy entirely

1. Install ngrok and authenticate with a free account token:
   ngrok config add-authtoken YOUR_TOKEN

2. Start the dashboard normally:
   streamlit run dashboard/app.py --server.headless true --server.port 8501

3. In a SEPARATE terminal, start the tunnel:
   ngrok http 8501

4. Use the "Forwarding" URL ngrok prints (e.g. https://xxxx.ngrok-free.dev),
   NOT Lightning's own port-forwarded URL.

5. Keep BOTH terminals running (the Streamlit server and the ngrok tunnel) —
   closing either one kills the dashboard's public access immediately.

Note: free-tier ngrok URLs may show an interstitial warning page on first
visit ("You are about to visit a site served by ngrok") — click "Visit Site"
to continue, this is normal, not an error.
