from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

ENABLED = os.environ.get('NS_V53_AXIS_NORMALIZE','1') != '0'
WRITE_AUDIT = os.environ.get('NS_V53_WRITE_AXIS_AUDIT','1') != '0'
STRIP_TITLES = os.environ.get('NS_V53_STRIP_TITLES','1') != '0'
FONT_FAMILY = 'Arial'

SPECS = {
    'standard': (58.0, 42.0),
    'wide': (70.0, 42.0),
    'heatmap': (46.0, 46.0),
    'forest': (60.0, 38.0),
    'histogram': (58.0, 40.0),
}

SEMANTIC_COLORS = {
    'teal': '#2A9D8F',
    'blue': '#38598C',
    'purple': '#8B7DAA',
    'orange': '#C1743C',
    'olive': '#90A86B',
    'rose': '#C97C7C',
    'slate': '#6B8FA8',
    'dark': '#4D4D4D',
    'mid': '#8C8C8C',
    'light': '#CFCFCF',
    'vlight': '#EEEEEE',
}

# old -> new palette remap, emphasizing the reference-like blue/teal/purple/orange range.
COLOR_REMAP = {
    '#168b8c': SEMANTIC_COLORS['teal'], '#2a9d8f': SEMANTIC_COLORS['teal'], '#4d908e': SEMANTIC_COLORS['teal'],
    '#3e6fa3': SEMANTIC_COLORS['blue'], '#4f789d': SEMANTIC_COLORS['blue'], '#4e79a7': SEMANTIC_COLORS['blue'],
    '#7568a9': SEMANTIC_COLORS['purple'], '#8174a8': SEMANTIC_COLORS['purple'], '#8a6aa5': SEMANTIC_COLORS['purple'], '#7b6d8d': SEMANTIC_COLORS['purple'],
    '#d49a3a': SEMANTIC_COLORS['orange'], '#c1833e': SEMANTIC_COLORS['orange'], '#c39a46': SEMANTIC_COLORS['orange'], '#d4a373': SEMANTIC_COLORS['orange'],
    '#3d8b7a': SEMANTIC_COLORS['olive'], '#5f9473': SEMANTIC_COLORS['olive'], '#8fa57a': SEMANTIC_COLORS['olive'],
    '#a94f63': SEMANTIC_COLORS['rose'], '#b67598': SEMANTIC_COLORS['rose'], '#b56576': SEMANTIC_COLORS['rose'], '#c97c7c': SEMANTIC_COLORS['rose'],
    '#6b9fb3': SEMANTIC_COLORS['slate'], '#577590': SEMANTIC_COLORS['slate'], '#6baed6': SEMANTIC_COLORS['slate'],
    '#264653': SEMANTIC_COLORS['dark'], '#355c7d': SEMANTIC_COLORS['blue'], '#355f7f': SEMANTIC_COLORS['blue'], '#2f4b7c': SEMANTIC_COLORS['blue'],
    '#c7654c': '#B86B4B', '#bc6c25': '#B86B4B', '#b86b4b': '#B86B4B',
    '#999999': SEMANTIC_COLORS['mid'], '#888888': SEMANTIC_COLORS['mid'], '#8c8c8c': SEMANTIC_COLORS['mid'],
    '#d9d9d9': SEMANTIC_COLORS['light'], '#cfcfcf': SEMANTIC_COLORS['light'], '#eeeeee': SEMANTIC_COLORS['vlight'],
    '#111111': '#111111', '#4d4d4d': '#4D4D4D', '#ffffff': '#FFFFFF',
}


def _hex(c):
    try:
        from matplotlib.colors import to_hex
        return to_hex(c, keep_alpha=False).lower()
    except Exception:
        return None


def _map_color(c):
    h = _hex(c)
    return COLOR_REMAP.get(h, c) if h else c


def _is_colorbar(ax):
    try:
        if ax.get_label() == '<colorbar>': return True
        if getattr(ax, '_colorbar', None) is not None: return True
    except Exception:
        pass
    return False


def _primary_axes(fig):
    axes=[ax for ax in fig.axes if getattr(ax,'get_visible',lambda:False)() and not _is_colorbar(ax)]
    if len(axes) <= 1: return axes
    unique=[]
    for ax in axes:
        p=ax.get_position().bounds
        if not any(max(abs(a-b) for a,b in zip(p,q.get_position().bounds)) < 0.02 for q in unique):
            unique.append(ax)
    return unique


def _classify(ax, fname=''):
    name=str(fname).lower()
    if any(k in name for k in ['heatmap','matrix','confusion']): return 'heatmap'
    if any(k in name for k in ['forest','coefficient','effect_ci','summary_forest']): return 'forest'
    if any(k in name for k in ['hist','null_distribution','permutation_null']): return 'histogram'
    if any(k in name for k in ['trajectory','profile','by_budget','curve','frequency_response','state_budget']): return 'wide'
    try:
        if len(getattr(ax,'images',[]))>0: return 'heatmap'
        ylabels=[t.get_text() for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        if len(ylabels)>=7 and max([len(x) for x in ylabels]+[0])>=8: return 'forest'
    except Exception:
        pass
    return 'standard'


def _sanitize_text(fig):
    if STRIP_TITLES:
        try:
            if getattr(fig,'_suptitle',None) is not None: fig._suptitle.set_text('')
        except Exception: pass
    for ax in fig.axes:
        if STRIP_TITLES:
            try: ax.set_title('')
            except Exception: pass
        try: ax.grid(False)
        except Exception: pass
        for label in [getattr(ax.xaxis,'label',None), getattr(ax.yaxis,'label',None)]:
            if label is not None:
                try: label.set_fontfamily(FONT_FAMILY); label.set_fontsize(6.4)
                except Exception: pass
        for t in list(ax.get_xticklabels())+list(ax.get_yticklabels()):
            try: t.set_fontfamily(FONT_FAMILY); t.set_fontsize(5.4)
            except Exception: pass
        try:
            ax.tick_params(axis='both', which='major', length=2.5, width=0.55, pad=2.0)
        except Exception: pass
        leg=ax.get_legend()
        if leg is not None:
            try: leg.set_frame_on(False)
            except Exception: pass
            for t in leg.get_texts():
                try: t.set_fontfamily(FONT_FAMILY); t.set_fontsize(5.4)
                except Exception: pass
        for s in ax.spines.values():
            try: s.set_linewidth(0.55)
            except Exception: pass

        # lines: force filled markers and lighter ribbons feel.
        for line in ax.get_lines():
            try: line.set_color(_map_color(line.get_color()))
            except Exception: pass
            try: line.set_linewidth(min(max(float(line.get_linewidth()),0.45),1.05))
            except Exception: pass
            try:
                mk = line.get_marker()
                if mk not in [None, 'None', ' ', '']:
                    c = line.get_color()
                    line.set_markersize(min(max(float(line.get_markersize()),3.2),4.6))
                    line.set_markerfacecolor(c)
                    line.set_markeredgecolor(c)
                    line.set_markeredgewidth(0.5)
                    line.set_alpha(0.95)
            except Exception: pass

        # scatter / collections
        for coll in ax.collections:
            try:
                arr = getattr(coll, 'get_array', lambda: None)()
                if arr is None:
                    from matplotlib.colors import to_rgba
                    fc = coll.get_facecolors(); ec = coll.get_edgecolors()
                    # if hollow, fill with edge color
                    if (fc is None or len(fc) == 0) and ec is not None and len(ec):
                        coll.set_facecolors(ec)
                        fc = coll.get_facecolors()
                    if fc is not None and len(fc):
                        new=[]
                        for c in fc:
                            mapped = _map_color(c)
                            rgba = list(to_rgba(mapped))
                            rgba[3] = max(0.80, c[3] if len(c)>3 else 0.90)
                            new.append(rgba)
                        coll.set_facecolors(new)
                    if ec is not None and len(ec):
                        new=[]
                        for c in ec:
                            mapped = _map_color(c)
                            rgba = list(to_rgba(mapped))
                            rgba[3] = max(0.90, c[3] if len(c)>3 else 0.95)
                            new.append(rgba)
                        coll.set_edgecolors(new)
                    try:
                        sizes = np.asarray(coll.get_sizes(), float)
                        if sizes.size:
                            coll.set_sizes(np.clip(sizes, 18, 42))
                    except Exception: pass
                    try:
                        lw = np.asarray(coll.get_linewidths(), float)
                        if lw.size:
                            coll.set_linewidths(np.clip(lw, 0.35, 0.8))
                    except Exception: pass
                else:
                    # likely heatmap or color-mapped scatter; keep data mapping but clamp edge clutter.
                    try:
                        coll.set_linewidths(0.0)
                    except Exception: pass
            except Exception: pass

        # ribbons / patches
        for patch in getattr(ax,'patches',[]):
            try:
                fc=patch.get_facecolor(); ec=patch.get_edgecolor()
                from matplotlib.colors import to_rgba
                mf=_map_color(fc); me=_map_color(ec)
                r=list(to_rgba(mf)); r[3]=min(0.14, fc[3] if len(fc)>3 else 0.14); patch.set_facecolor(r)
                r=list(to_rgba(me)); r[3]=ec[3] if len(ec)>3 else r[3]; patch.set_edgecolor(r)
            except Exception: pass

        # heatmap images
        try:
            import numpy as _np
            for im in getattr(ax, 'images', []):
                data = im.get_array()
                if data is None: continue
                arr = _np.asarray(data, float)
                if _np.nanmin(arr) < 0 and _np.nanmax(arr) > 0:
                    from matplotlib.colors import LinearSegmentedColormap
                    cmap = LinearSegmentedColormap.from_list('ns_div', ['#2166AC', '#F7F7F7', '#B2182B'], N=256)
                    im.set_cmap(cmap)
                else:
                    from matplotlib.colors import LinearSegmentedColormap
                    cmap = LinearSegmentedColormap.from_list('ns_seq', ['#F7FBFF', '#6BAED6', '#08519C'], N=256)
                    im.set_cmap(cmap)
            # annotated heatmaps often live in texts
            for txt in ax.texts:
                try:
                    txt.set_fontfamily(FONT_FAMILY)
                    txt.set_fontsize(min(float(txt.get_fontsize()), 6.0))
                except Exception: pass
        except Exception:
            pass

        # clamp huge annotation text (e.g. 4F statistics)
        for txt in ax.texts:
            try:
                txt.set_fontfamily(FONT_FAMILY)
                txt.set_fontsize(min(float(txt.get_fontsize()), 6.0))
            except Exception: pass
    for txt in fig.texts:
        try:
            txt.set_fontfamily(FONT_FAMILY)
            txt.set_fontsize(min(float(txt.get_fontsize()), 6.0))
        except Exception: pass


def _axis_decor_in(ax, renderer, dpi):
    bb=ax.get_window_extent(renderer)
    try: tb=ax.get_tightbbox(renderer)
    except Exception: tb=bb
    if tb is None: tb=bb
    return {
        'left':max(0.0,(bb.x0-tb.x0)/dpi), 'right':max(0.0,(tb.x1-bb.x1)/dpi),
        'bottom':max(0.0,(bb.y0-tb.y0)/dpi), 'top':max(0.0,(tb.y1-bb.y1)/dpi),
    }


def _normalize_single_axis(fig, fname=''):
    axes=_primary_axes(fig)
    if len(axes)!=1: return None
    ax=axes[0]
    kind=_classify(ax,fname)
    tw_mm,th_mm=SPECS[kind]; tw=tw_mm/25.4; th=th_mm/25.4
    try:
        fig.canvas.draw(); renderer=fig.canvas.get_renderer(); dpi=fig.dpi
        dec=_axis_decor_in(ax,renderer,dpi)
    except Exception:
        dec={'left':0.45,'right':0.12,'bottom':0.35,'top':0.12}
    pad=1.8/25.4
    cbs=[a for a in fig.axes if _is_colorbar(a)]
    cb=None
    if len(cbs)==1:
        p=cbs[0].get_position()
        if p.height > p.width*2: cb=cbs[0]
    if cb is None:
        fw=dec['left']+tw+dec['right']+2*pad
        fh=dec['bottom']+th+dec['top']+2*pad
        fw=max(fw,tw+0.25); fh=max(fh,th+0.22)
        fig.set_size_inches(fw,fh,forward=True)
        ax.set_position([(dec['left']+pad)/fw,(dec['bottom']+pad)/fh,tw/fw,th/fh])
    else:
        try:
            fig.canvas.draw(); renderer=fig.canvas.get_renderer(); dpi=fig.dpi
            cbd=_axis_decor_in(cb,renderer,dpi)
        except Exception: cbd={'left':0,'right':0.22,'bottom':0,'top':0}
        cbw=2.8/25.4; gap=3.2/25.4
        fw=dec['left']+tw+gap+cbw+cbd['right']+2*pad
        fh=max(dec['bottom']+th+dec['top']+2*pad, th+0.25)
        fig.set_size_inches(fw,fh,forward=True)
        x0=(dec['left']+pad)/fw; y0=(dec['bottom']+pad)/fh
        ax.set_position([x0,y0,tw/fw,th/fh])
        cb.set_position([x0+tw/fw+gap/fw,y0,cbw/fw,th/fh])
    try: fig.canvas.draw()
    except Exception: pass
    return {'family':kind,'target_axis_width_mm':tw_mm,'target_axis_height_mm':th_mm}


def _audit(fig, fname, norm_meta):
    try:
        axes=_primary_axes(fig); rows=[]
        fw,fh=fig.get_size_inches()
        for i,ax in enumerate(axes):
            p=ax.get_position()
            rows.append({'axis_index':i,'axis_width_mm':round(fw*p.width*25.4,3),'axis_height_mm':round(fh*p.height*25.4,3),'x0':p.x0,'y0':p.y0,'w_frac':p.width,'h_frac':p.height})
        return {'file':str(fname),'figure_width_mm':round(fw*25.4,3),'figure_height_mm':round(fh*25.4,3),'normalization':norm_meta,'axes':rows}
    except Exception as e:
        return {'file':str(fname),'error':repr(e)}

_PATCHED=False

def install():
    global _PATCHED
    if _PATCHED or not ENABLED: return
    try:
        from matplotlib.figure import Figure
    except Exception: return
    orig=Figure.savefig
    def savefig(self, fname, *args, **kwargs):
        _sanitize_text(self)
        meta=_normalize_single_axis(self,fname)
        kwargs['bbox_inches']=None
        out=orig(self,fname,*args,**kwargs)
        if WRITE_AUDIT:
            try:
                p=Path(fname)
                side=p.with_suffix(p.suffix+'.axis.json')
                side.write_text(json.dumps(_audit(self,fname,meta),indent=2,ensure_ascii=False,default=lambda x: float(x) if hasattr(x,'item') else str(x)),encoding='utf-8')
            except Exception as e:
                try: print('[V5.3 axis sidecar warning]', repr(e))
                except Exception: pass
        return out
    Figure.savefig=savefig
    _PATCHED=True

install()
