import argparse
import json
import os

import numpy as np
import pandas as pd


class PseudoMCSampling:
    def __init__(self, linefailprob, wind_farms_output, random_failure_prob, options):
        self.linefailprob, self.line_labels, self.time_labels = self._prepare_matrix(linefailprob, "linefailprob")
        self.wind_farms_output, self.wind_farm_labels, wind_time_labels = self._prepare_matrix(
            wind_farms_output,
            "wind_farms_output"
        )
        self.random_failure_prob = self._prepare_failure_prob(random_failure_prob)
        self.options = options or {}

        self.alt_time_labels = None
        if self.time_labels and wind_time_labels:
            if len(wind_time_labels) != len(self.time_labels):
                raise ValueError(
                    "Wind farm output and line failure probability horizons differ in length."
                )
            if wind_time_labels != self.time_labels:
                self.alt_time_labels = {
                    'lines': self.time_labels,
                    'wind_farms': wind_time_labels
                }
        if not self.time_labels:
            self.time_labels = wind_time_labels

        self.ng = self.random_failure_prob['probgen'].shape[0]
        self.nl = self.linefailprob.shape[0]
        self.nw = self.wind_farms_output.shape[0]
        self.T = self.linefailprob.shape[1]

        self.system_state = None
        self.component_failure = []
        self.faulty_line_counts = []
        self.metadata = {
            'line_labels': self.line_labels,
            'time_labels': self.time_labels,
            'wind_farm_labels': self.wind_farm_labels,
            'generator_count': self.ng,
            'alternate_time_labels': self.alt_time_labels
        }

    @classmethod
    def from_excel(cls, linefail_path, random_failure_path, wind_output_path, sheet_name, options,
                   line_index_col=0, wind_index_col=0):
        linefail_df = pd.read_excel(
            linefail_path,
            sheet_name=sheet_name,
            index_col=line_index_col,
            engine="openpyxl",
        )
        wind_df = pd.read_excel(
            wind_output_path,
            sheet_name=sheet_name,
            index_col=wind_index_col,
            engine="openpyxl",
        )
        probgen_df = pd.read_excel(random_failure_path, sheet_name='probgen', header=None, engine="openpyxl")
        probbr_df = pd.read_excel(random_failure_path, sheet_name='probbr', header=None, engine="openpyxl")
        random_failure_prob = {
            'probgen': probgen_df.to_numpy(dtype=float).flatten(),
            'probbr': probbr_df.to_numpy(dtype=float).flatten()
        }
        return cls(linefail_df, wind_df, random_failure_prob, options)

    @staticmethod
    def _prepare_matrix(data, name):
        index = list(data.index) if hasattr(data, "index") else None
        columns = list(data.columns) if hasattr(data, "columns") else None
        matrix = data.to_numpy(dtype=float) if hasattr(data, "to_numpy") else np.asarray(data, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")
        return matrix, index, columns

    @staticmethod
    def _to_vector(values, name):
        if values is None:
            raise ValueError(f"{name} is required.")
        if hasattr(values, "to_numpy"):
            array = values.to_numpy(dtype=float)
        else:
            array = np.asarray(values, dtype=float)
        return array.flatten()

    def _prepare_failure_prob(self, random_failure_prob):
        if not isinstance(random_failure_prob, dict):
            raise TypeError("random_failure_prob must be a dict with 'probgen' and 'probbr'.")
        return {
            'probgen': self._to_vector(random_failure_prob.get('probgen'), 'probgen'),
            'probbr': self._to_vector(random_failure_prob.get('probbr'), 'probbr')
        }

    def simulate(self):
        batch_size = int(self.options.get('batch_size', 1))
        consider_random_failures = bool(self.options.get('consider_random_failures', 1))
        consider_repair = bool(self.options.get('consider_repair', 0))
        T_re = int(self.options.get('T_re', 0))
        show_progress = bool(self.options.get('show_progress', False))
        progress_updates = int(self.options.get('progress_updates', 10))
        progress_updates = max(1, progress_updates)
        prefix = self.options.get('progress_prefix', '')
        prefix = f"{prefix} " if prefix else ''
        progress_interval = max(1, batch_size // progress_updates)

        self.system_state = {
            'generators': np.zeros((batch_size, self.ng, self.T), dtype=int),
            'lines': np.zeros((batch_size, self.nl, self.T), dtype=int),
            'wind_farms': np.zeros((batch_size, self.nw, self.T), dtype=float),
            'metadata': self.metadata
        }
        self.component_failure = []
        self.faulty_line_counts = []

        probgen = self.random_failure_prob['probgen']
        line_random_prob = np.zeros(self.nl, dtype=float)
        base_probbr = self.random_failure_prob['probbr']
        copy_length = min(base_probbr.size, self.nl)
        if copy_length:
            line_random_prob[:copy_length] = base_probbr[:copy_length]

        for iteration in range(batch_size):
            generator_status = np.zeros((self.ng, self.T), dtype=int)
            line_status = np.zeros((self.nl, self.T), dtype=int)
            generator_repair_status = np.zeros((self.ng, self.T), dtype=int)
            line_repair_status = np.zeros((self.nl, self.T), dtype=int)

            if consider_random_failures:
                for g in range(self.ng):
                    for t in range(self.T):
                        if t > 0 and generator_status[g, t - 1] == 1:
                            generator_status[g, t] = 1
                        elif probgen[g] > 0 and np.random.rand() < probgen[g]:
                            generator_status[g, t] = 1

                        if consider_repair and T_re > 0 and t >= T_re:
                            if generator_status[g, t - T_re:t].sum() == T_re:
                                generator_repair_status[g, t] = 1
                        if generator_repair_status[g, t] == 1:
                            generator_status[g, t] = 0

                for l in range(self.nl):
                    for t in range(self.T):
                        if t > 0 and line_status[l, t - 1] == 1:
                            line_status[l, t] = 1
                        elif line_random_prob[l] > 0 and np.random.rand() < line_random_prob[l]:
                            line_status[l, t] = 1

                        if consider_repair and T_re > 0 and t >= T_re:
                            if line_status[l, t - T_re:t].sum() == T_re:
                                line_repair_status[l, t] = 1
                        if line_repair_status[l, t] == 1:
                            line_status[l, t] = 0

            wind_farm_status = self.wind_farms_output.copy()

            for t in range(self.T):
                for l in range(self.nl):
                    if t > 0 and line_status[l, t - 1] == 1:
                        line_status[l, t] = 1
                        continue
                    if line_status[l, t] == 0 and self.linefailprob[l, t] > 0:
                        if np.random.rand() < self.linefailprob[l, t]:
                            line_status[l, t] = 1

            generator_failure_rows = np.any(generator_status == 1, axis=1)
            transmission_line_failure_rows = np.any(line_status == 1, axis=1)
            faulty_line_count = int(np.sum(transmission_line_failure_rows))

            self.system_state['generators'][iteration] = generator_status
            self.system_state['lines'][iteration] = line_status
            self.system_state['wind_farms'][iteration] = wind_farm_status

            self.component_failure.append({
                'generators': generator_failure_rows,
                'transmission_lines': transmission_line_failure_rows,
                'line_labels': self.line_labels
            })
            self.faulty_line_counts.append(faulty_line_count)

            if show_progress and ((iteration + 1) % progress_interval == 0 or iteration == batch_size - 1):
                print(f"{prefix}Monte Carlo progress: {iteration + 1}/{batch_size}", flush=True)

    def get_results(self):
        return self.system_state, self.component_failure, self.faulty_line_counts


def _default_path(filename):
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, filename)


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Run pseudo Monte Carlo sampling for hurricane scenarios.")
    parser.add_argument("--linefail", default=_default_path("impact_assessment_simplified.xlsx"),
                        help="Path to line failure probability workbook.")
    parser.add_argument("--random", default=_default_path("random_failure_prob.xlsx"),
                        help="Path to random failure probability workbook.")
    parser.add_argument("--wind", default=_default_path("wind_farms_output.xlsx"),
                        help="Path to wind farm output workbook.")
    parser.add_argument("--sheet", default="all",
                        help="Sheet name(s) shared across the workbooks; use comma list or 'all'.")
    parser.add_argument("--batch", type=int, default=1000, help="Number of Monte Carlo samples.")
    parser.add_argument("--repair", action="store_true", help="Enable repair logic with window T_re.")
    parser.add_argument("--tre", type=int, default=24, help="Repair window length in hours.")
    parser.add_argument("--no-random", action="store_true", help="Disable baseline random failures.")
    parser.add_argument("--dump-json", default="", help="Optional path to dump per-sheet summaries as JSON.")
    parser.add_argument("--output", default=_default_path("mc_simulation_results.xlsx"),
                        help="Path of the Excel workbook to write simulation results to.")
    return parser.parse_args()
def _resolve_sheet_names(linefail_path, wind_path, sheet_arg):
    if sheet_arg is None:
        raise ValueError("Sheet argument must be provided.")

    line_book = pd.ExcelFile(linefail_path, engine="openpyxl")
    wind_book = pd.ExcelFile(wind_path, engine="openpyxl")
    line_sheets = line_book.sheet_names
    wind_sheets = wind_book.sheet_names
    wind_sheet_set = set(wind_sheets)
    shared = [name for name in line_sheets if name in wind_sheet_set]

    if not shared:
        raise ValueError("No common sheet names found between line failure and wind output workbooks.")

    sheet_arg = str(sheet_arg).strip()

    if sheet_arg.lower() == 'all':
        return shared

    requested = [name.strip() for name in sheet_arg.split(',') if name.strip()]
    if not requested:
        raise ValueError("No valid sheet names were supplied.")

    missing_line = [name for name in requested if name not in line_sheets]
    missing_wind = [name for name in requested if name not in wind_sheet_set]
    if missing_line or missing_wind:
        messages = []
        if missing_line:
            messages.append(f"{linefail_path}: {missing_line}")
        if missing_wind:
            messages.append(f"{wind_path}: {missing_wind}")
        raise ValueError("Requested sheet(s) not found -> " + "; ".join(messages))

    return requested


def _build_component_df(prefix, array, labels, time_labels, dtype=None):
    data = np.asarray(array)
    if dtype is not None:
        data = data.astype(dtype, copy=False)

    if data.ndim != 3:
        raise ValueError("Component array must be 3-dimensional (batch, count, time).")

    batch_size, component_count, horizon = data.shape
    sample_labels = [f"Sample_{i + 1}" for i in range(batch_size)]

    if labels is None:
        labels = [f"{prefix}{i + 1}" for i in range(component_count)]
    else:
        labels = [str(label).strip() for label in labels]
        if prefix == "Line_":
            normalized: list[str] = []
            for idx, label in enumerate(labels, start=1):
                if not label:
                    normalized.append(f"{prefix}{idx}")
                elif label.startswith(prefix):
                    normalized.append(label)
                else:
                    normalized.append(f"{prefix}{label}")
            labels = normalized

    if len(labels) != component_count:
        raise ValueError("Component label length does not match the data dimensions.")

    column_labels = time_labels or [f"Hour_{i + 1}" for i in range(horizon)]
    if len(column_labels) != horizon:
        raise ValueError("Time label length does not match horizon length.")

    reshaped = data.reshape(batch_size * component_count, horizon)
    index = pd.MultiIndex.from_product([sample_labels, labels], names=['Sample', 'Component'])
    return pd.DataFrame(reshaped, index=index, columns=column_labels)


def _write_sheet(writer, sheet_name, sampler, system_state, faulty_counts):
    sheet_title = str(sheet_name) if sheet_name else "Sheet"
    sheet_title = sheet_title[:31]

    start_row = 0
    batch_size = system_state['generators'].shape[0]
    summary_rows = [
        ("batch_size", batch_size),
        ("line_count", sampler.nl),
        ("generator_count", sampler.ng),
        ("wind_farm_count", sampler.nw),
        ("time_horizon", sampler.T)
    ]
    summary_df = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])
    summary_df.to_excel(writer, sheet_name=sheet_title, index=False, startrow=start_row)
    start_row += len(summary_df) + 2

    sample_labels = [f"Sample_{i + 1}" for i in range(batch_size)]
    faulty_df = pd.DataFrame({
        'Sample': sample_labels,
        'FaultyLineCount': faulty_counts
    })
    faulty_df.to_excel(writer, sheet_name=sheet_title, index=False, startrow=start_row)
    start_row += len(faulty_df) + 2

    lines_df = _build_component_df('Line_', system_state['lines'], sampler.line_labels, sampler.time_labels, dtype=int)
    lines_df.to_excel(writer, sheet_name=sheet_title, startrow=start_row)
    start_row += len(lines_df) + 2

    generators_df = _build_component_df('Gen_', system_state['generators'], None, sampler.time_labels, dtype=int)
    generators_df.to_excel(writer, sheet_name=sheet_title, startrow=start_row)
    start_row += len(generators_df) + 2

    wind_df = _build_component_df('Farm_', system_state['wind_farms'], sampler.wind_farm_labels,
                                  sampler.time_labels, dtype=float)
    wind_df.to_excel(writer, sheet_name=sheet_title, startrow=start_row)
    start_row += len(wind_df) + 2

    if sampler.alt_time_labels:
        alt_df = pd.DataFrame(sampler.alt_time_labels)
        alt_df.to_excel(writer, sheet_name=sheet_title, startrow=start_row)


def _run_cli():
    args = _parse_cli_args()
    options = {
        'batch_size': args.batch,
        'consider_random_failures': 0 if args.no_random else 1,
        'consider_repair': 1 if args.repair else 0,
        'T_re': args.tre
    }
    sheet_names = _resolve_sheet_names(args.linefail, args.wind, args.sheet)

    total_sheets = len(sheet_names)
    print(f"Preparing to simulate {total_sheets} typhoon scenario(s)...")

    summaries = []
    json_payloads = []

    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        for idx, sheet in enumerate(sheet_names, start=1):
            sheet_options = dict(options)
            sheet_options['show_progress'] = True
            sheet_options['progress_prefix'] = f"[{idx}/{total_sheets}][{sheet}]"
            sampler = PseudoMCSampling.from_excel(
                linefail_path=args.linefail,
                random_failure_path=args.random,
                wind_output_path=args.wind,
                sheet_name=sheet,
                options=sheet_options
            )
            print(f"[{idx}/{total_sheets}] Simulating typhoon sheet '{sheet}' (batch={args.batch})...", flush=True)
            sampler.simulate()
            system_state, component_failure, faulty_counts = sampler.get_results()

            _write_sheet(writer, sheet, sampler, system_state, faulty_counts)

            summary_entry = {
                'sheet': str(sheet),
                'line_count': sampler.nl,
                'generator_count': sampler.ng,
                'wind_farm_count': sampler.nw,
                'time_horizon': sampler.T,
                'faulty_mean': float(np.mean(faulty_counts)) if faulty_counts else 0.0,
                'faulty_min': int(np.min(faulty_counts)) if faulty_counts else 0,
                'faulty_max': int(np.max(faulty_counts)) if faulty_counts else 0,
                'faulty_first_samples': faulty_counts[:10]
            }
            summaries.append(summary_entry)

            print(f"[{idx}/{total_sheets}] Completed '{sheet}' -> faulty_mean={summary_entry['faulty_mean']:.2f}, "
                  f"faulty_range=[{summary_entry['faulty_min']}, {summary_entry['faulty_max']}]", flush=True)

            if args.dump_json:
                json_payloads.append({
                    'sheet': str(sheet),
                    'batch_size': args.batch,
                    'time_horizon': sampler.T,
                    'line_count': sampler.nl,
                    'generator_count': sampler.ng,
                    'wind_farm_count': sampler.nw,
                    'faulty_line_counts': faulty_counts,
                    'metadata': system_state['metadata']
                })

    if args.dump_json and json_payloads:
        with open(args.dump_json, "w", encoding="utf-8") as fh:
            json.dump(json_payloads, fh, ensure_ascii=False, indent=2)
        print(f"Summary written to {args.dump_json}")

    print("Simulation finished.")
    print(f"Saved workbook to {args.output}")
    for entry in summaries:
        print(f"  Sheet {entry['sheet']}: lines={entry['line_count']}, horizon={entry['time_horizon']}h, "
              f"faulty_mean={entry['faulty_mean']:.2f}, "
              f"faulty_range=[{entry['faulty_min']}, {entry['faulty_max']}], "
              f"first10={entry['faulty_first_samples']}")


if __name__ == "__main__":
    _run_cli()