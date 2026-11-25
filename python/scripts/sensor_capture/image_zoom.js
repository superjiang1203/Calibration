function mountImageZoom(content, image) {
    const viewport = document.createElement('div');
    viewport.style.cssText = 'overflow:auto;max-height:70vh;touch-action:pan-x pan-y;background:#080e15;border-radius:6px';
    const toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex;justify-content:center;align-items:center;gap:6px;flex-wrap:wrap;margin:6px 0 2px';
    content.before(toolbar, viewport);
    viewport.append(content);
    viewport.after(toolbar);
    content.style.cssText += ';display:block;position:relative;max-width:none;margin:0 auto';
    image.style.cssText += ';display:block;width:100%;height:auto;max-width:none;max-height:none;margin:0 auto';
    const level = document.createElement('output');
    level.setAttribute('aria-live', 'polite');
    let scale = 1;
    function fitWidth() {
        return Math.min(viewport.clientWidth,
            viewport.clientHeight ? window.innerHeight * 0.68 * image.naturalWidth / image.naturalHeight : Infinity,
            image.naturalWidth);
    }
    function layout() {
        if (image.naturalWidth && viewport.clientWidth) {
            content.style.width = fitWidth() * scale + 'px';
        }
        level.textContent = Math.round(scale * 100) + '%';
        viewport.style.cursor = scale > 1 ? 'grab' : 'zoom-in';
    }
    function change(next, x = viewport.clientWidth / 2, y = viewport.clientHeight / 2) {
        const old = scale;
        const box = viewport.getBoundingClientRect();
        const prior = content.getBoundingClientRect();
        const dx = x + box.left - prior.left;
        const dy = y + box.top - prior.top;
        scale = Math.max(1, Math.min(8, next));
        if (scale === old) return;
        layout();
        const nextBox = content.getBoundingClientRect();
        viewport.scrollLeft += dx * scale / old + nextBox.left - box.left - x;
        viewport.scrollTop += dy * scale / old + nextBox.top - box.top - y;
    }
    for (const [label, action] of [['−', () => change(scale / 1.25)],
        ['+', () => change(scale * 1.25)], ['适应画面', () => change(1)]]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = label;
        button.onclick = action;
        toolbar.append(button);
    }
    toolbar.append(level);
    viewport.addEventListener('wheel', event => {
        if (!image.naturalWidth) return;
        event.preventDefault();
        const box = viewport.getBoundingClientRect();
        change(scale * (event.deltaY < 0 ? 1.15 : 1 / 1.15),
            event.clientX - box.left, event.clientY - box.top);
    }, {passive: false});
    image.draggable = false;
    let drag;
    viewport.addEventListener('pointerdown', event => {
        if (event.pointerType !== 'mouse' || event.button !== 0 || scale === 1) return;
        drag = [event.clientX, event.clientY, viewport.scrollLeft, viewport.scrollTop];
        viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener('pointermove', event => {
        if (!drag) return;
        viewport.scrollLeft = drag[2] - (event.clientX - drag[0]);
        viewport.scrollTop = drag[3] - (event.clientY - drag[1]);
    });
    viewport.addEventListener('lostpointercapture', () => { drag = null; });
    image.addEventListener('load', layout);
    const observer = new ResizeObserver(layout);
    observer.observe(viewport);
    layout();
    return () => observer.disconnect();
}
