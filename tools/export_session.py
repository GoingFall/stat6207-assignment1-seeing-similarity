"""Export the current OpenCode session without rebuilding historical pilot archives."""
import argparse,json,subprocess
from pipelines.prepare_data import ROOT

parser=argparse.ArgumentParser();parser.add_argument('--cli',default='opencode');args=parser.parse_args();cli=args.cli
result=subprocess.run([cli,'session','export','ses_f4122ee06ffdO2N65t2ateSQyq'],capture_output=True,check=True)
folder=ROOT/'docs/session';folder.mkdir(parents=True,exist_ok=True)
data=json.loads(result.stdout);(folder/'session-transcript.json').write_bytes(result.stdout)
lines=['# OpenCode session transcript','','Snapshot before final archive construction. Full tool details are in the JSON export.','']
for message in data['messages']:
    if message.get('type')=='user':lines.extend(['## User','',message.get('text',''),''])
    elif message.get('type')=='assistant':
        for part in message.get('content',[]):
            if part.get('type')=='text':lines.extend(['## Assistant','',part.get('text',''),''])
            elif part.get('type')=='tool':lines.extend(['Tool: '+part.get('tool',part.get('name','tool call')),''])
(folder/'session-transcript.md').write_text('\n'.join(lines),encoding='utf-8')
print('Exported',len(data['messages']),'messages')
