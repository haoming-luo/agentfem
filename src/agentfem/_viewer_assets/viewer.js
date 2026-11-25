/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0 */
// Keep the live websocket document in place when downloading an export.
window.AgentFEMViewer = {
  install(app) {
    app.component('viewer-axes', {
      props: ['camera', 'dark', 'select'],
      data: () => ({targets: []}),
      template: `<div style="position:relative;width:100px;height:100px">
        <canvas ref="canvas" width="200" height="200" style="width:100px;height:100px" aria-hidden="true"></canvas>
        <button v-for="a in targets" :key="a.label" class="viewer-axis-button"
          :style="{left:a.x+'px',top:a.y+'px',color:a.color}" :title="'View along '+a.label"
          :aria-label="'View along '+a.label" @click="select(a.label)">{{a.label}}</button>
        </div>`,
      mounted() {
        const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
        const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
        const unit = a => { const n=Math.hypot(...a); return n ? a.map(v=>v/n) : [0,0,1]; };
        let previous='';
        this.stopped=false;
        const draw = () => {
          if (this.stopped) return;
          const camera = this.camera?.();
          const key = JSON.stringify([camera,this.dark]);
          if (camera?.position && camera?.focalPoint && key!==previous) {
            previous=key;
            const z=unit(camera.position.map((v,i)=>v-camera.focalPoint[i]));
            const x=unit(cross(camera.viewUp,z)), y=cross(z,x);
            const colors=this.dark ? ['#eda6b9','#9ed4c4','#a8bee9'] : ['#983f60','#267762','#3d5c9b'];
            const ctx=this.$refs.canvas.getContext('2d');
            ctx.clearRect(0,0,200,200); ctx.save();ctx.scale(2,2);
            const axes=[0,1,2].map(i=>({i,dx:x[i]*29,dy:-y[i]*29,depth:z[i]})).sort((a,b)=>a.depth-b.depth);
            ctx.lineCap='round';ctx.lineJoin='round';ctx.font='600 12px -apple-system, Arial';ctx.textAlign='center';ctx.textBaseline='middle';
            const targets=[];
            for (const a of axes) {
              const length=Math.hypot(a.dx,a.dy), ex=50+a.dx,ey=50+a.dy;
              ctx.strokeStyle=colors[a.i];ctx.fillStyle=colors[a.i];ctx.lineWidth=2.2;
              ctx.beginPath();ctx.moveTo(50,50);ctx.lineTo(ex,ey);ctx.stroke();
              const ux=length>1 ? a.dx/length : 0,uy=length>1 ? a.dy/length : -1;
              if(length>5){ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-ux*5-uy*2.2,ey-uy*5+ux*2.2);ctx.lineTo(ex-ux*5+uy*2.2,ey-uy*5-ux*2.2);ctx.closePath();ctx.fill();}
              targets.push({label:'XYZ'[a.i],x:ex+ux*10,y:ey+uy*10,color:colors[a.i]});
            }
            this.targets=targets;
            ctx.fillStyle=this.dark?'#ced9e6':'#8292a5';ctx.beginPath();ctx.arc(50,50,2.5,0,2*Math.PI);ctx.fill();ctx.restore();
          }
          this.raf=requestAnimationFrame(draw);
        };
        draw();
      },
      beforeUnmount(){this.stopped=true;cancelAnimationFrame(this.raf);}
    });
    app.config.globalProperties.$viewer = {
      navigate(address) {
        const url = new URL(address);
        if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1') return;
        window.location.assign(url.href);
      },
      async download(relative, filename, mime) {
        const url = new URL(relative, window.location.href);
        if (url.origin !== window.location.origin || !url.pathname.startsWith('/viewer-exports/')) {
          throw new Error('Select a completed local export.');
        }
        const response = await window.fetch(url);
        if (!response.ok) throw new Error('Export is unavailable. Please export again.');
        const bytes = await response.blob();
        const objectUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
        const anchor = document.createElement('a');
        anchor.href = objectUrl;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
      },
    };
  },
};
