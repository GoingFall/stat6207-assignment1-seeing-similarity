"""Real Edge interactions, image loading, map click and mobile overflow."""
import json,threading
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
from pipelines.prepare_data import ROOT,OUT

def main():
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(ROOT/'site/v1.0')))
    threading.Thread(target=server.serve_forever,daemon=True).start();errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge');page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Locked data loaded')")
        assert page.locator('#nearest figure').count()==5 and page.locator('#farthest figure').count()==5
        assert page.locator('#comparisons tr').count()==12
        assert page.locator('#metric option[value="l2"]').inner_text()=='L2 (raw features)'
        assert page.locator('#metric option[value="cosine"]').inner_text()=='Cosine (unit directions)'
        page.locator('#normalization-guide summary').click()
        assert 'rank mismatches: 0' in page.locator('#normalization-verification').inner_text()
        page.locator('#normalization-guide').screenshot(path=str(R1/'normalization-guide.png'))
        data=json.loads((ROOT/'site/v1.0/assets/data.json').read_text())
        page.wait_for_function("document.querySelectorAll('#heatmap-output canvas').length===2")
        before=page.locator('#nearest').inner_text();page.select_option('#metric','l1')
        assert page.locator('#nearest').inner_text()!=before
        for model in ['resnet18','dinov2','clip']:
            page.select_option('#model',model)
            for metric in ['l1','l2','cosine']:
                page.select_option('#metric',metric)
                assert ('unnormalized' if metric in ['l1','l2'] else 'unit-normalized') in page.locator('#distance-note').inner_text()
                for condition in data['protocol']['conditions']:
                    page.select_option('#condition',condition);page.select_option('#query','499');page.select_option('#k','9')
                    key=f'{model}|{metric}|{condition}';rank=data['ranks'][key][499]
                    expected=[data['reference'][i]['id'] for i in rank[:5]]
                    assert page.locator('#nearest figure').evaluate_all('(items)=>items.map(x=>x.dataset.id)')==expected
                    dogs=sum(data['reference'][i]['source_label']=='dog' for i in rank[:9])
                    assert f'{dogs} dog votes, {9-dogs} cat votes' in page.locator('#vote').inner_text()
                    assert page.locator('#voters li').count()==9
                    assert page.locator('#lab').get_attribute('data-configuration')==key+'|499|9'
                    assert condition in page.locator('#vote').inner_text()
        assert page.locator('#heatmap-toggle').is_disabled()
        assert page.locator('#heatmap-output canvas').count()==0
        page.wait_for_function("document.querySelector('#map-dinov2')._fullLayout !== undefined")
        page.evaluate("document.querySelector('#map-dinov2').emit('plotly_click',{points:[{customdata:4}]})")
        assert page.locator('#query').input_value()=='4'
        page.select_option('#model','resnet18');page.select_option('#metric','cosine');page.select_option('#condition','clean');page.select_option('#k','5')
        page.wait_for_function("document.querySelector('#heatmap-output .heatmap-grid')?.dataset.query==='cat_0504'")
        page.locator('#farthest button').nth(2).click()
        assert page.locator('#heatmap-pair').input_value()=='7'
        page.wait_for_function("document.querySelector('#heatmap-status').textContent.includes('farthest #3')")
        page.locator('#heatmap-toggle').uncheck();assert page.locator('#heatmap-output canvas').count()==0
        page.locator('#heatmap-toggle').check()
        for condition in data['protocol']['conditions']:
            for metric in ['l1','l2','cosine']:
                page.select_option('#condition',condition);page.select_option('#metric',metric)
                page.wait_for_function("document.querySelector('#heatmap-output canvas') !== null")
                assert 'Unable' not in page.locator('#heatmap-status').inner_text()
        assert page.locator('#case-query option').count()==14
        for value in page.locator('#case-query option').evaluate_all('(items)=>items.map(x=>x.value)'):
            page.select_option('#case-query',value);ident=data['queries'][int(value)]['id']
            page.wait_for_function('(id)=>document.querySelector("#case-heatmaps .heatmap-grid")?.dataset.query===id',arg=ident)
            assert page.locator('#case-heatmaps canvas').count()==2
        page.select_option('#case-pair','4')
        page.wait_for_function("document.querySelector('#case-pair-status').textContent.includes('nearest #5')")
        page.locator('#inspect-case').click();assert page.locator('#query').input_value()==page.locator('#case-query').input_value()
        assert data['queries'][int(page.locator('#query').input_value())]['id'] in page.locator('#map-selection').inner_text()
        page.wait_for_function("document.querySelector('#heatmap-output canvas') !== null")
        page.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
        page.locator('#maps').screenshot(path=str(R1/'umap-after.png'))
        page.locator('#lab').screenshot(path=str(R1/'retrieval-heatmaps.png'))
        page.locator('#cases').screenshot(path=str(R1/'case-heatmaps.png'))
        page.screenshot(path=str(R1/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.wait_for_function('document.documentElement.scrollWidth <= innerWidth+1')
        page.locator('#normalization-guide').screenshot(path=str(R1/'normalization-mobile.png'))
        page.screenshot(path=str(R1/'mobile.png'),full_page=True)
        assert not errors,errors;browser.close()
    server.shutdown();(R1/'browser_check.json').write_text(json.dumps(dict(status='passed',browser='Edge',page_errors=errors,checks=['63 control combinations match recorded ranks/votes','actual k voting set','21 condition/metric heatmaps','14 unseen/error heatmaps','pair selector and overlay toggle','model scope guard','12 comparison rows','map click and selected marker','all image resources','390px responsive layout']),indent=2))
    print('Formal browser checks passed')

if __name__=='__main__':main()
