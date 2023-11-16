import json
import pandas as pd


def create_clinical_record_map(json_files):
    cr_map = {'nan': ""}
    for c in json_files:
        with open(c) as json_file:
            c = json.load(json_file)
            cr_id = c['Clinical Trial ID']
        cr_map[cr_id] = format_record(c)
    return cr_map


def format_record(cr):
    x = []
    for k in ['Intervention', 'Results', 'Adverse Events', 'Eligibility']:
        x += cr[k]
    return "\n".join(x)


def create_dataset(json_file, cr_json_files):
    cr_map = create_clinical_record_map(cr_json_files)

    df = pd.read_json(json_file).T
    df.Label = df.Label.map({"Contradiction": 0, "Entailment": 1})
    df['Primary Inp'] = df['Primary_id'].map(lambda x: cr_map[x])
    df['Secondary Inp'] = df['Secondary_id'].map(lambda x: cr_map[str(x)])
    df['Inp'] = df["Statement"] + "\n" + df['Primary Inp'] + df['Secondary Inp']
    df = df.drop(["Primary Inp", "Secondary Inp", "Primary_id", "Secondary_id", "Type", "Statement"], axis=1)
    return df
