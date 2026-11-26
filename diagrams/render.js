const puppeteer=require('puppeteer-core');const fs=require('fs');const path=require('path');
(async()=>{
 const mx=fs.readFileSync(path.join(__dirname,'../node_modules/mxgraph/javascript/mxClient.min.js'),'utf8');
 const b=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',args:['--no-sandbox']});
 for(const f of process.argv.slice(2)){
  const xml=fs.readFileSync(f,'utf8');
  const p=await b.newPage(); await p.setViewport({width:1700,height:1200,deviceScaleFactor:3});
  await p.setContent(`<html><body style="margin:0;background:#fff"><div id="g" style="position:absolute;left:0;top:0;"></div><script>window.mxLoadResources=false;window.mxLoadStylesheets=false;window.mxBasePath='.';</script></body></html>`);
  await p.addScriptTag({content:mx});
  const box=await p.evaluate((xml)=>{
   const doc=mxUtils.parseXml(xml);const model=doc.getElementsByTagName('mxGraphModel')[0];
   const g=new mxGraph(document.getElementById('g'));g.setHtmlLabels(true);
   new mxCodec(model.ownerDocument).decode(model,g.getModel());
   const bb=g.getGraphBounds();g.view.setTranslate(-bb.x+20,-bb.y+20);
   const c=document.getElementById('g');c.style.width=(bb.width+40)+'px';c.style.height=(bb.height+40)+'px';
   return {w:Math.ceil(bb.width+40),h:Math.ceil(bb.height+40)};
  },xml);
  await p.setViewport({width:box.w,height:box.h,deviceScaleFactor:3});
  await p.screenshot({path:f.replace(/\.drawio$/,'.png'),clip:{x:0,y:0,width:box.w,height:box.h}});
  console.log(f,box);
 }
 await b.close();})();
