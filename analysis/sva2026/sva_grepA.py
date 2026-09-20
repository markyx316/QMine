import sys,re,html,subprocess,io
def fetch(url):
    r=subprocess.run(['curl','-sL','--compressed','--max-time','60','-A','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',url],capture_output=True)
    return r.stdout
def to_text(b):
    if b[:4]==b'%PDF':
        from pypdf import PdfReader
        rd=PdfReader(io.BytesIO(b)); return '\n'.join((p.extract_text() or '') for p in rd.pages), len(rd.pages)
    s=b.decode('utf-8','ignore')
    s=re.sub(r'<script.*?</script>|<style.*?</style>','',s,flags=re.S)
    t=html.unescape(re.sub(r'<[^>]+>',' ',s)); return re.sub(r'\s+',' ',t), None
if __name__=='__main__':
    url,out=sys.argv[1],sys.argv[2]; kws=sys.argv[3:]
    b=open(url,'rb').read() if url.startswith('/') else fetch(url)
    t,n=to_text(b)
    open(out,'w').write(t)
    print('URL',url,'bytes',len(b),'chars',len(t),'pages',n)
    for kw in kws:
        hits=[m.start() for m in re.finditer(re.escape(kw),t)]
        print(f'--- {kw}: {len(hits)} hits')
        for h in hits[:3]:
            print('   ',t[max(0,h-150):h+150].replace('\n',' '))
