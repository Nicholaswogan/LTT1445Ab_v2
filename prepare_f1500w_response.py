"""Export a photon response from local Pandeia reference files (no downloads)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from astropy.io import fits


def export(root, output):
    miri = root/'jwst/miri'
    telescope = root/'jwst/telescope'
    config = json.loads((miri/'config.json').read_text())
    tel_config = json.loads((telescope/'config.json').read_text())
    files = [miri/config['paths']['f1500w'],
             miri/config['paths']['miri_imager_qe'],
             telescope/tel_config['paths']['otefile']]
    curves = []
    for path in files:
        with fits.open(path) as f:
            curves.append((np.array(f[1].data['WAVELENGTH']), np.array(f[1].data['THROUGHPUT'])))
    w, filt = curves[0]
    positive = np.flatnonzero(filt > 0)
    # Keep all nonzero wings and an adjacent zero at each end.
    sl = slice(max(0, positive[0]-1), min(len(w), positive[-1]+2))
    w, response = w[sl], filt[sl].copy()
    for wave, throughput in curves[1:]:
        if w[0] < wave[0] or w[-1] > wave[-1]:
            raise ValueError('Pandeia component does not cover filter support')
        response *= np.interp(w, wave, throughput)
    # MIRI mirror efficiencies are wavelength independent in these reference data;
    # their normalization and collecting area cancel in an eclipse ratio.
    np.savetxt(output, np.c_[w, response], header='wavelength_um photon_response\nF1500W filter/instrument transmission x imager QE x OTE throughput')
    metadata = {'components': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
                'method': 'product of filter/instrument transmission, imager QE and OTE throughput; linear interpolation to filter grid',
                'support_um': [float(w[0]), float(w[-1])],
                'reference_data_version': config['-delivered-for-version-'],
                'note': 'Local reference data; verify calibration choice with the observer. No aperture or time-dependent correction.'}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pandeia_data', type=Path)
    parser.add_argument('--output', type=Path, default=Path('inputs/F1500W_response.txt'))
    args = parser.parse_args()
    export(args.pandeia_data, args.output)
