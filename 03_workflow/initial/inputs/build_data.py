"""Prepare the July single-setpoint EnergyPlus data for the 17R8C model.

Minute inputs retain terminal discharge temperature, moist-air specific heat,
the four boundary temperatures, and dry-air infiltration conductance. July 1
initializes the continuous model; July 2-22 supplies calibration data and
July 23-31 supplies model-selection validation data. States remain continuous
across the partition boundaries."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
TARGET='THERMAL ZONE: SPACE 103 LAB B'
EXPECTED_REFERENCE_SHA256='712258c194bd62da6908ba9449bc51d2fcfc1d9f2c21eb6cdc25c4ece5db2980'
EXPECTED_CORE_IDF_SHA256='15a0330f618f6a97ba83423f25285e6ad7431663017ca55e8c24d92d8e24dce2'
SOURCE_UPLOADS={
    'HIBO_MCMC_1min.csv':'HIBO_LabB_July_Single_Setpoint_Excitation.csv',
    'HIBO.idf':'HIBO_LabB_July_Single_Setpoint_Excitation.idf',
    'simulation_audit.json':'Audit of supplied single-setpoint SQL and CSV',
}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):digest.update(b)
    return digest.hexdigest()

def core_idf_sha(text):
    # Compare geometry, material, and HVAC objects outside the excitation configuration.
    ignored={'runperiod','schedule:compact','thermostatsetpoint:dualsetpoint','zonecontrol:thermostat','thermostatsetpoint:singleheatingorcooling','output:variable'}
    rows=[]
    for chunk in re.sub(r'!.*','',text).split(';'):
        if not chunk.strip():continue
        a=[x.strip().casefold()for x in chunk.split(',')]
        if a[0] not in ignored:rows.append(a)
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),len(rows)

def main():
    assert sha(HERE/'reference.json')==EXPECTED_REFERENCE_SHA256,'Reference file changed: revalidate physical model.'
    core_hash,n_core=core_idf_sha((HERE/'HIBO.idf').read_text())
    assert core_hash==EXPECTED_CORE_IDF_SHA256,'IDF geometry/material/HVAC core changed: rebuild reference.'
    header=pd.read_csv(HERE/'HIBO_MCMC_1min.csv',nrows=0).columns
    names={c.strip().casefold():c for c in header};used={}
    requested=[
      'Date/Time',
      TARGET+':Zone Air Temperature [C](TimeStep)',
      TARGET+':Zone Total Internal Visible Radiation Heating Rate [W](TimeStep)',
      TARGET+':Zone Total Internal Radiant Heating Rate [W](TimeStep)',
      TARGET+':Zone Total Internal Convective Heating Rate [W](TimeStep)',
      TARGET+':Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)',
      TARGET+' SUPPLY INLET:System Node Mass Flow Rate [kg/s](TimeStep)',
      TARGET+' SUPPLY INLET:System Node Temperature [C](TimeStep)',
      'DXVAV SYS 1 SUPPLY FAN OUTLET:System Node Humidity Ratio [kgWater/kgDryAir](TimeStep)',
      TARGET+':Zone Infiltration Mass Flow Rate [kg/s](TimeStep)',
      'THERMAL ZONE: PLENUM:Zone Air Temperature [C](TimeStep)',
      'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
      'THERMAL ZONE: SPACE 102 - PLUS:Zone Air Temperature [C](TimeStep)',
      'Environment:Site Ground Temperature [C](TimeStep)',
      'Environment:Site Day Type Index [](TimeStep)',
      TARGET+':Zone Thermostat Control Type [](TimeStep)',
      TARGET+':Zone Thermostat Heating Setpoint Temperature [C](TimeStep)',
      TARGET+':Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)',
      # Independent audit; not used to calculate the fitted forcing arrays.
      TARGET+':Zone People Sensible Heating Rate [W](TimeStep)',
      TARGET+':Zone Electric Equipment Total Heating Rate [W](TimeStep)',
    ]
    assert all(c.casefold()in names for c in requested),'Required minute output column is missing.'
    f=pd.read_csv(HERE/'HIBO_MCMC_1min.csv',usecols=[names[c.casefold()]for c in requested])
    def col(name):
        c=names[name.casefold()];used[name]=c
        return f[c].to_numpy(float)
    def zone(name):return col(TARGET+':'+name+'(TimeStep)')
    # Reconstruct equipment gains before clipping small negative occupant gains.
    vis=zone('Zone Total Internal Visible Radiation Heating Rate [W]')
    rad=zone('Zone Total Internal Radiant Heating Rate [W]')
    conv=zone('Zone Total Internal Convective Heating Rate [W]')
    light=vis/.2
    people=(rad-.7*light)/.3
    equip=conv-.1*light-.7*people
    assert min(light.min(),people.min(),equip.min())>-1e-7
    people=np.maximum(people,0.)
    dt=f[names['date/time']].str.extract(r'\s*(\d+)/(\d+)\s+(\d+):(\d+):(\d+)').astype(int)
    month=dt[0].to_numpy();day=dt[1].to_numpy()
    minute_end=dt[2].to_numpy()*60+dt[3].to_numpy()
    assert len(f)==44640 and (month==7).all() and (dt[4]==0).all()
    assert np.array_equal((day-1)*1440+minute_end-1,np.arange(44640)), 'Missing, duplicated, or misordered end-of-interval minute.'
    assert not f[names['date/time']].duplicated().any()
    assert f[names['date/time']].iloc[0].strip()=='07/01  00:01:00'
    assert f[names['date/time']].iloc[-1].strip()=='07/31  24:00:00'
    # The IDF RunPeriod calendar starts July 1 on Sunday.
    day_type=col('Environment:Site Day Type Index [](TimeStep)').astype(np.int64)
    expected_day_type=((day-1)%7)+1
    assert np.array_equal(day_type,expected_day_type),'CSV day types differ from the Sunday-start source calendar.'
    weekday=(day_type>=2)&(day_type<=6)
    # Equipment density and floor area are taken from the IDF.
    schedule=np.select([minute_end<=480,minute_end<=540,minute_end<=840,minute_end<=900,minute_end<=960,minute_end<=1080,minute_end<=1260],[.02,.4,.9,.8,.7,.5,.3],default=.02)
    schedule=np.where(weekday,schedule,.02)
    expected_equip=18.413809881984*6.78*schedule
    equipment_deviation=float(np.max(abs(expected_equip-equip)))
    assert equipment_deviation<1e-6,'Recovered sensible equipment differs from the unchanged IDF schedule.'
    people_deviation=float(np.max(abs(people-zone('Zone People Sensible Heating Rate [W]'))))
    direct_equip_deviation=float(np.max(abs(equip-zone('Zone Electric Equipment Total Heating Rate [W]'))))
    assert people_deviation<1e-6 and direct_equip_deviation<1e-6
    mass=col(TARGET+' SUPPLY INLET:System Node Mass Flow Rate [kg/s](TimeStep)')
    supply=col(TARGET+' SUPPLY INLET:System Node Temperature [C](TimeStep)')
    hum=col('DXVAV SYS 1 SUPPLY FAN OUTLET:System Node Humidity Ratio [kgWater/kgDryAir](TimeStep)')
    heating=zone('Zone Thermostat Heating Setpoint Temperature [C]')
    cooling=zone('Zone Thermostat Cooling Setpoint Temperature [C]')
    assert np.array_equal(np.unique((heating+cooling)/2),np.array([20.,22.,24.,26.,28.]))
    assert (cooling==heating).all(), 'Single heating/cooling target required'
    assert (zone('Zone Thermostat Control Type []')==3.).all(), 'Control type must be 3 throughout July'
    train=(day>=2)&(day<=22);valid=day>=23;conditioning=day==1
    assert not (train&valid).any() and not (train&conditioning).any() and (train|valid|conditioning).all()
    d=dict(y=zone('Zone Air Temperature [C]'),boundaries=np.column_stack([
      col('THERMAL ZONE: PLENUM:Zone Air Temperature [C](TimeStep)'),
      col('Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'),
      col('THERMAL ZONE: SPACE 102 - PLUS:Zone Air Temperature [C](TimeStep)'),
      col('Environment:Site Ground Temperature [C](TimeStep)')]),
      g_hvac=mass*(1005+1860*hum),tsupply=supply,
      ginf=zone('Zone Infiltration Mass Flow Rate [kg/s]')*1005,
      gains=np.ascontiguousarray(np.column_stack([zone('Zone Windows Total Transmitted Solar Radiation Rate [W]'),light,people,equip])),
      train=train,valid=valid,month=month,day=day,dt_seconds=60.,
      conditioning=conditioning,weekday=weekday,minute_end=minute_end,day_type=day_type,
      time_seconds=np.arange(1,len(f)+1,dtype=np.int64)*60,
      heating_setpoint=heating,cooling_setpoint=cooling)
    assert all(np.isfinite(a).all()for a in d.values())
    assert mass.min()>=0 and hum.min()>=0 and d['ginf'].min()>=0
    np.savez_compressed(HERE/'data.npz',**d)
    simulation_audit=json.loads((HERE/'simulation_audit.json').read_text())
    assert simulation_audit['csv_sha256']==sha(HERE/'HIBO_MCMC_1min.csv')
    assert simulation_audit['full_month_coverage_verified'] and simulation_audit['no_severe_or_fatal_errors_in_sql']
    audit=dict(
      source_upload_names=SOURCE_UPLOADS,source_period='July 1-31, end-of-interval minute data, IDF Sunday-start local standard-time calendar',
      target_zone=TARGET,rows=len(f),dt_seconds=60,first_raw_timestamp=f[names['date/time']].iloc[0],last_raw_timestamp=f[names['date/time']].iloc[-1],
      conditioning_minutes=int(conditioning.sum()),train_minutes=int(train.sum()),validation_minutes=int(valid.sum()),
      partition={'conditioning':'July 1, first 24 h; no likelihood or validation score','training':'July 2-22','validation':'July 23-31'},
      continuous_simulation='All 44,640 minutes are simulated from the first E+ zone air temperature, with no state reset at partition boundaries.',
      reference_sha256=EXPECTED_REFERENCE_SHA256,reference_role='Original IDF-derived broad prior reference values, not fixed R or C estimates.',
      unchanged_IDF_core={'objects':n_core,'sha256':core_hash,'excluded_intentionally_modified_types':['RunPeriod','Schedule:Compact','ThermostatSetpoint:DualSetpoint','ZoneControl:Thermostat','ThermostatSetpoint:SingleHeatingOrCooling','Output:Variable']},
      boundary_order=['plenum temperature','outdoor dry-bulb temperature','Space 102 - Plus air temperature','site ground temperature'],
      gain_order=['solar_transmitted','lighting','occupant_sensible','equipment_sensible'],
      gain_recovery={'lighting':'visible / 0.2','occupant_sensible_raw':'(radiant - 0.7*lighting) / 0.3','equipment_sensible':'convective - 0.1*lighting - 0.7*occupant_sensible_raw','occupant_sensible':'maximum(occupant_sensible_raw,0), after equipment reconstruction'},
      hvac_conductance='Lab B supply-inlet mass flow * (1005 + 1860 * AHU supply-fan-outlet humidity ratio)',
      terminal_temperature='Lab B SUPPLY INLET System Node Temperature, downstream of terminal reheat. AHU SAT is not substituted.',
      infiltration_conductance='Lab B infiltration mass flow * 1005',
      equipment_IDF_schedule_max_difference_W=equipment_deviation,
      people_direct_output_max_difference_W=people_deviation,equipment_direct_output_max_difference_W=direct_equip_deviation,
      central_setpoint_levels_C=[20,22,24,26,28],heating_cooling_band_width_C=0,thermostat_control_type=3,
      EnergyPlus_run=simulation_audit,
      used_columns=used,
      hashes={p.name:sha(p)for p in sorted(HERE.iterdir())if p.is_file()and not p.name.startswith('.')and p.name!='metadata.json'},
      array_shapes={k:list(np.asarray(v).shape)for k,v in d.items()},
    )
    (HERE/'metadata.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:audit[k]for k in ['rows','conditioning_minutes','train_minutes','validation_minutes','equipment_IDF_schedule_max_difference_W','people_direct_output_max_difference_W','equipment_direct_output_max_difference_W','reference_sha256']},indent=2))
if __name__=='__main__':main()
