"""Render complete current/previous PDFs for visual comparison and record layout checks."""
import json
import pymupdf
from PIL import Image,ImageOps,ImageDraw
from pipelines.prepare_data import ROOT,OUT

def main():
    folder=OUT/'results/report_review';folder.mkdir(exist_ok=True)
    reports={'pilot':ROOT/'backup/pilot/report.pdf','previous':ROOT/'backup/v3-before-reorganization/report.pdf','current':ROOT/'report.pdf'}
    summary={}
    for name,path in reports.items():
        if not path.exists():continue
        doc=pymupdf.open(path);tiles=[];texts=[];issues=[]
        destination=folder if name=='current' else path.parent/'report_review'
        destination.mkdir(exist_ok=True)
        target=destination/name;target.mkdir(exist_ok=True)
        for i,page in enumerate(doc):
            pix=page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5));pix.save(str(target/f'page-{i+1:02}.png'))
            im=Image.open(target/f'page-{i+1:02}.png').convert('RGB');im.thumbnail((298,422))
            tile=Image.new('RGB',(318,456),'#dddddd');tile.paste(im,((318-im.width)//2,20));ImageDraw.Draw(tile).text((12,438),f'{name} page {i+1}',fill='black');tiles.append(tile)
            texts.append(page.get_text())
            for b in page.get_text('blocks'):
                if b[0]<0 or b[1]<0 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1:issues.append(dict(page=i+1,bounds=b[:4]))
        for start in range(0,len(tiles),4):
            subset=tiles[start:start+4];sheet=Image.new('RGB',(636,456*((len(subset)+1)//2)),'white')
            for j,tile in enumerate(subset):sheet.paste(tile,((j%2)*318,(j//2)*456))
            sheet.save(destination/f'{name}-contact-{start//4+1}.png')
        (target/'text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
        summary[name]=dict(pages=len(doc),characters=sum(map(len,texts)),page_characters=list(map(len,texts)),out_of_bounds=issues)
    current=pymupdf.open(ROOT/'report.pdf');content='\n'.join(page.get_text() for page in current)
    assert not summary['current']['out_of_bounds']
    assert all(section in content for section in ['A1.','A2.','A3.','A4.','A5.','B1.','B2.','B3.','C. OpenCode'])
    assert len(current.get_toc())==11
    summary['checks']=['all current text bounds','all eight rubric tasks','11 section bookmarks']
    (folder/'layout_check.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
