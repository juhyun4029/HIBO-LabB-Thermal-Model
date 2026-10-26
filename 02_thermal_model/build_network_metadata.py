"""Export the reference network, prior table, and topology diagram."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math

HERE = Path(__file__).resolve().parent
BOUNDARIES = [
    {"index": 0, "id": "T_plenum", "description": "Thermal Zone: Plenum air temperature"},
    {"index": 1, "id": "T_outdoor", "description": "Site outdoor air dry-bulb temperature"},
    {"index": 2, "id": "T_neighbor", "description": "Thermal Zone: Space 102 - Plus air temperature"},
    {"index": 3, "id": "T_ground", "description": "Site ground temperature"},
]
GAIN_SOURCES = ["solar", "lighting", "occupant", "equipment"]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def diagram(network, destination):
    def text(x, y, value, size=14, fill="#182f40", anchor="middle", weight="normal"):
        return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{fill}" font-weight="{weight}">{html.escape(value)}</text>'

    def resistor(x1, x2, y, label):
        middle = (x1 + x2) / 2
        xs = [middle - 48 + 8 * i for i in range(13)]
        pts = [(x1, y), (xs[0], y)] + [(x, y + (7 if i % 2 else -7)) for i, x in enumerate(xs[1:-1], 1)] + [(xs[-1], y), (x2, y)]
        return f'<polyline points="{" ".join(f"{x},{yy}" for x, yy in pts)}" fill="none" stroke="#284b63" stroke-width="2"/>' + text(middle, y - 14, label, 12)

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="1040" viewBox="0 0 1240 1040" role="img" aria-labelledby="title desc">',
           '<title id="title">HIBO Lab B initial 17R8C thermal network</title>',
           '<desc id="desc">Seven capacitive envelope nodes, three massless window paths, and one zone-air capacitance. Four boundary temperatures are prescribed.</desc>',
           '<rect width="1240" height="1040" fill="white"/>',
           '<g font-family="Arial, Helvetica, sans-serif">',
           text(620, 38, 'HIBO Lab B · 17R8C', 25, weight='bold'),
           text(620, 69, '17 resistances = 7 × 2 + 3   |   8 capacitances = 7 envelope + zone air', 16),
           text(110, 112, 'Prescribed boundary', 15, weight='bold'),
           text(525, 112, 'Envelope state', 15, weight='bold'),
           text(1095, 112, 'Zone-air state', 15, weight='bold'),
           '<line x1="970" y1="151" x2="970" y2="889" stroke="#284b63" stroke-width="3"/>']
    for index, component in enumerate(network['components']):
        y = 157 + 80 * index
        boundary = BOUNDARIES[component['boundary']]['id']
        svg.append(f'<rect x="22" y="{y-21}" width="176" height="42" rx="7" fill="#edf3f8" stroke="#aabcc8"/>')
        svg.append(text(110, y+5, boundary, 15))
        if component['kind'] == 'massive':
            svg.append(resistor(198, 440, y, component['name'] + ':R1'))
            svg.append(f'<rect x="440" y="{y-26}" width="170" height="52" rx="8" fill="#e4f1f5" stroke="#126b99" stroke-width="2"/>')
            svg.append(text(525, y-3, component['name'], 14, weight='bold'))
            svg.append(text(525, y+16, component['name'] + ':C', 12))
            svg.append(resistor(610, 970, y, component['name'] + ':R2'))
        else:
            svg.append(resistor(198, 970, y, component['name'] + ':R (massless)'))
        svg.append(f'<circle cx="970" cy="{y}" r="4" fill="#284b63"/>')
    svg.extend(['<line x1="970" y1="450" x2="1010" y2="450" stroke="#284b63" stroke-width="3"/>',
                '<rect x="1010" y="410" width="192" height="80" rx="9" fill="#fceadd" stroke="#b85e28" stroke-width="2"/>',
                text(1106, 442, 'T_zone', 20, weight='bold'), text(1106, 468, 'zone_air:C', 15),
                text(620, 956, 'Four heat-source allocation vectors act on all 8 states.', 16),
                text(620, 984, 'HVAC and infiltration heat enter the zone-air balance.', 16),
                text(620, 1012, 'Repeated boundary labels denote the same prescribed temperature, not additional states.', 13, '#536b7a'), '</g></svg>'])
    destination.write_text('\n'.join(svg) + '\n', encoding='utf-8')


def main():
    reference_path = HERE / 'reference.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    components = reference['components']
    resistances = []
    capacitances = []
    for component in components:
        boundary = BOUNDARIES[component['boundary']]['id']
        if component['kind'] == 'massive':
            state = 'T_' + component['name']
            resistances.extend([
                {'parameter': component['name'] + ':R1', 'node_a': boundary, 'node_b': state, 'reference_K_per_W': component['R1']},
                {'parameter': component['name'] + ':R2', 'node_a': state, 'node_b': 'T_zone', 'reference_K_per_W': component['R2']},
            ])
            capacitances.append({'parameter': component['name'] + ':C', 'state': state, 'reference_J_per_K': component['C']})
        else:
            resistances.append({'parameter': component['name'] + ':R', 'node_a': boundary, 'node_b': 'T_zone', 'reference_K_per_W': component['R']})
    capacitances.append({'parameter': 'zone_air:C', 'state': 'T_zone', 'reference_J_per_K': reference['Czone']})
    network = {
        'name': 'HIBO_LabB_17R8C', 'target_zone': reference['target_zone'],
        'n_R': len(resistances), 'n_C': len(capacitances),
        'boundary_temperatures': BOUNDARIES,
        'states': [c['state'] for c in capacitances],
        'components': components, 'resistances': resistances, 'capacitances': capacitances,
        'heat_sources': GAIN_SOURCES,
        'heat_allocation': {'shape': [4, 8], 'order': [c['name'] for c in components[:7]] + ['zone_air'], 'constraint': 'Each heat-source allocation vector sums to 1.'},
        'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'window_host_role': 'Geometric association only; does not restrict reduction candidate generation.',
    }
    write_json(HERE / 'network.json', network)
    write_csv(HERE / 'resistances.csv', resistances)
    write_csv(HERE / 'capacitances.csv', capacitances)
    rows = []
    for component in components:
        rows.append({'component': component['name'], 'kind': component['kind'],
                     'boundary_temperature': BOUNDARIES[component['boundary']]['id'],
                     'idf_surfaces': '; '.join(component['surfaces']),
                     'area_m2': component['area'], 'gross_area_m2': component.get('gross_area', ''),
                     'n_R': 2 if component['kind'] == 'massive' else 1,
                     'n_C': 1 if component['kind'] == 'massive' else 0,
                     'R_total_K_per_W': component['R'], 'R1_K_per_W': component.get('R1', ''),
                     'R2_K_per_W': component.get('R2', ''), 'C_J_per_K': component['C']})
    write_csv(HERE / 'components.csv', rows)
    thermal = reference['physical_parameters'] + [{'name': 'zone_air:C', 'unit': 'J/K', 'reference': reference['Czone']}]
    prior_rows = [{'parameter': row['name'], 'unit': row['unit'], 'distribution': 'lognormal',
                   'physical_mean': row['reference'], 'physical_sd': 2 * row['reference'],
                   'log_mean': math.log(row['reference']) - .5 * math.log(5),
                   'log_sd': math.sqrt(math.log(5)), 'coefficient_of_variation': 2.0,
                   'mean_basis': 'IDF properties and declared RC reference conventions',
                   'sd_basis': 'Chosen broad prior coefficient of variation, not an IDF uncertainty output'} for row in thermal]
    write_csv(HERE / 'prior_parameters.csv', prior_rows)
    write_json(HERE / 'priors.json', {
        'independent_parameters': 55, 'reported_values': 59,
        'thermal': {'independent_count': 25, 'distribution': 'lognormal', 'physical_mean': 'reference', 'physical_sd': '2 * reference', 'log_sd': math.sqrt(math.log(5)), 'log_mean': 'log(reference) - log(5)/2'},
        'heat_allocation': {'sources': GAIN_SOURCES, 'states': network['heat_allocation']['order'], 'distribution_per_source': 'Dirichlet', 'concentration_per_state': [1] * 8, 'independent_count': 28, 'reported_count': 32},
        'noise': {'independent_count': 2, 'log_sigma_hour': {'distribution': 'truncated_normal', 'mean': math.log(.3), 'sd': 1., 'lower': -5., 'upper': 2.}, 'eta': {'distribution': 'truncated_normal', 'mean': 1.5, 'sd': 1.5, 'lower': -5., 'upper': 6.}, 'rho_hour': '0.98 * sigmoid(eta)', 'phi_5min': 'rho_hour ** (5/60)', 'sigma_5min': 'sigma_hour * sqrt((1 - phi_5min**2)/(1 - rho_hour**2))'},
        'priors_not_derived_from_idf': ['thermal coefficient of variation', 'Dirichlet concentration', 'noise prior parameters'],
    })
    diagram(network, HERE / 'thermal_network_17R8C.svg')
    print(f'Exported {len(resistances)} resistances, {len(capacitances)} capacitances, and {len(prior_rows)} thermal priors.')


if __name__ == '__main__':
    main()
