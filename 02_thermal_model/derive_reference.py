"""Derive the 17R8C reference model from the archived EnergyPlus IDF."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import numpy as np
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
DEFAULT_IDF = REPOSITORY / "03_workflow" / "initial" / "inputs" / "HIBO.idf"
ARCHIVED_REFERENCE = REPOSITORY / "03_workflow" / "initial" / "inputs" / "reference.json"
def objects(path):
    txt=re.sub(r'!.*','',Path(path).read_text())
    return [[x.strip() for x in a.split(',')] for a in txt.split(';') if a.strip()]
def vertices(o,fen=False):return np.array(list(map(float,o[10 if fen else 12:]))).reshape(-1,3)
def area(v):return float(np.linalg.norm(np.cross(v,np.roll(v,-1,axis=0)).sum(axis=0))/2)

def derive_reference(idf_path):
    by={}
    for o in objects(idf_path):by.setdefault(o[0].lower(),{})[o[1].lower()]=o
    ss=by['buildingsurface:detailed'];fs=by['fenestrationsurface:detailed'];cons=by['construction'];layers=[];parts=[]
    def material(name):
        for typ in ['material','material:airgap','material:nomass','windowmaterial:glazing','windowmaterial:simpleglazingsystem']:
            if name.lower() not in by.get(typ,{}):continue
            o=by[typ][name.lower()]
            if typ=='material':
                t,k,rho,cp=map(float,o[3:7]);return dict(material=name,t=t,k=k,rho=rho,cp=cp,R_areal=t/k,C_areal=t*rho*cp)
            if typ in ['material:airgap','material:nomass']:
                return dict(material=name,t=0,k=0,rho=0,cp=0,R_areal=float(o[2 if typ.endswith('airgap') else 3]),C_areal=0)
            if typ=='windowmaterial:glazing':
                t=float(o[4]);k=float(o[15]);return dict(material=name,t=t,k=k,rho=0,cp=0,R_areal=t/k,C_areal=0)
            return dict(material=name,t=0,k=0,rho=0,cp=0,R_areal=1/float(o[2]),C_areal=0,includes_films=True)
        raise KeyError(name)
    def patch(component,name,construction,a,boundary,kind='opaque'):
        lay=[material(n) for n in cons[construction.lower()][2:]]
        rl=sum(x['R_areal'] for x in lay);ca=sum(x['C_areal'] for x in lay)
        ri=.13;ro=.04 if boundary==1 else .13
        if boundary==3:ro=0. # Standard Ground prescribes the outside-face temperature; no soil/outer film.
        if kind=='window' and any(x.get('includes_films') for x in lay):ri=0;ro=0
        R1=(.5*rl+ro)/a;R2=(.5*rl+ri)/a
        for x in lay:layers.append(dict(component=component,object=name,construction=construction,area=a,**x))
        row=dict(component=component,object=name,construction=construction,area=a,R1=R1,R2=R2,R=R1+R2,C=ca*a,kind=kind,boundary=boundary)
        parts.append(row);return row
    names=['ceiling','extwall_1','extwall_2','extwall_3','intwall_1','intwall_2','floor','extwindow_1','extwindow_2','intwindow_1']
    mappings=[(['Surface 8','Surface 71'],0),(['Surface 69 Ext','Surface 69 Low CMU'],1),(['Surface 70 Ext'],1),(['Surface 3 Ext'],1),(['Surface 5','Surface 6'],2),(['Surface 4'],2),(['Surface 1'],3)]
    window_groups={7:['HIBO CWC LabB Vision'],8:['HIBO CWA LabB West Lower CTIG','HIBO CWA LabB West Upper CTIG'],9:['HIBO Type A LabB-Control CTG A','HIBO Door 103 CTG Lite A']}
    comp=[]
    for i,(surfaces,boundary) in enumerate(mappings):
        pp=[];vv=[];gross=0
        for name in surfaces:
            s=ss[name.lower()];v=vertices(s);vv.append(v.tolist());ga=area(v);gross+=ga
            openings=[f for f in fs.values() if f[4].lower()==name.lower()]
            net=ga-sum(area(vertices(f,True)) for f in openings)
            if name=='Surface 69 Ext':
                fv=vertices(fs['hibo cwc labb vision'],True);fw=float(by['windowproperty:frameanddivider']['hibo cwc vision frame'][2]);W=np.ptp(fv[:,0]);H=np.ptp(fv[:,2]);net-=(W+2*fw)*(H+2*fw)-W*H
            assert net>0,(name,net)
            pp.append(patch(names[i],name,s[3],net,boundary))
            # Opaque leaf/frame strips belong to the host opaque block. No independent Door state.
            for f in openings:
                if f[2].lower()=='door':pp.append(patch(names[i],f[1],f[3],area(vertices(f,True)),boundary,'opaque_opening'))
        gs=np.array([1/p['R'] for p in pp]);R=1/gs.sum();f=np.sum(gs*np.array([p['R1']/p['R'] for p in pp]))/gs.sum()
        comp.append(dict(name=names[i],surfaces=surfaces,boundary=boundary,area=sum(p['area'] for p in pp),gross_area=gross,R=R,R1=R*f,R2=R*(1-f),C=sum(p['C'] for p in pp),vertices=vv,kind='massive',preaggregation='sum C, sum steady-state conductance, conductance-weighted R1 fraction'))
    for i,wlist in window_groups.items():
        boundary=1 if i<9 else 2;pp=[]
        for name in wlist:
            f=fs[name.lower()];pp.append(patch(names[i],name,f[3],area(vertices(f,True)),boundary,'window'))
        # CWC FrameAndDivider effect is an approximate prior correction. Frame lies outside the glazing rectangle.
        frameG=0.;frameA=0.
        if i==7:
            f=fs[wlist[0].lower()];v=vertices(f,True);W=np.ptp(v[:,0]);H=np.ptp(v[:,2]);fr=by['windowproperty:frameanddivider'][f[7].lower()]
            fw=float(fr[2]);dw=float(fr[11]);nh=int(fr[12]);nv=int(fr[13]);frameA=(W+2*fw)*(H+2*fw)-W*H
            divA=dw*(nh*W+nv*H)-nh*nv*dw*dw
            # Convert conductance excluding surface films to simple U with .17 m2K/W films.
            uf=1/(1/float(fr[5])+.17);ud=1/(1/float(fr[16])+.17)
            frameG=frameA*uf+divA*(ud-1.36)
        comp.append(dict(name=names[i],surfaces=wlist,boundary=boundary,area=sum(p['area'] for p in pp),R=1/(sum(1/p['R'] for p in pp)+frameG),C=0.,kind='massless',host={7:1,8:2,9:4}[i],frame_G_correction=frameG,frame_area=frameA))

    target='Thermal Zone: Space 103 LAB B'
    for c in comp[:7]:
        for surface in c['surfaces']:
            assert ss[surface.lower()][4]==target
    meta=dict(target_zone=target,components=comp,Czone=1.2*1005*comp[6]['area']*3.048,dt_seconds=60.,n_R=17,n_C_estimated=8,n_C_fixed=0,n_parameters=55,n_R_free=17,n_error_parameters=2,n_alpha_reported=32,n_alpha_independent=28)
    rows=[]
    for c in comp[:7]:
        for r in ['R1','R2']: rows.append(dict(name=c['name']+':'+r,unit='K/W',reference=c[r],role='prior_reference_only'))
    for c in comp[7:]: rows.append(dict(name=c['name']+':R',unit='K/W',reference=c['R'],role='prior_reference_only'))
    for c in comp[:7]: rows.append(dict(name=c['name']+':C',unit='J/K',reference=c['C'],role='prior_reference_only'))
    assert len(rows)==24
    meta['physical_parameters']=rows
    return meta, parts, layers


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf", type=Path, default=DEFAULT_IDF)
    parser.add_argument("--reference", type=Path, default=ARCHIVED_REFERENCE)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--check", action="store_true", help="Check the archived reference without writing files.")
    args = parser.parse_args()
    reference, patches, layers = derive_reference(args.idf)
    encoded = json.dumps(reference, indent=2).encode("utf-8")
    archived = args.reference.read_bytes()
    comparison = {
        "idf_sha256": hashlib.sha256(args.idf.read_bytes()).hexdigest(),
        "derived_reference_sha256": hashlib.sha256(encoded).hexdigest(),
        "archived_reference_sha256": hashlib.sha256(archived).hexdigest(),
        "byte_identical_reference": encoded == archived,
        "components": len(reference["components"]),
        "resistances": reference["n_R"],
        "capacitances": reference["n_C_estimated"],
        "material_patch_rows": len(patches),
        "material_layer_rows": len(layers),
        "method": "IDF geometry/material calculation using the archived reference construction conventions.",
    }
    if not comparison["byte_identical_reference"]:
        raise ValueError("Derived reference differs from the archived calibration reference.")
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "reference.json").write_bytes(encoded)
        write_csv(args.output / "idf_patches.csv", patches)
        write_csv(args.output / "idf_layers.csv", layers)
        (args.output / "derivation_validation.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
