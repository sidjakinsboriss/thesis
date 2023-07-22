from math import isnan

import pandas as pd

from analysis import TAGS


def _flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}


def _is_architectural(label):
    return label and 'not-ak' not in label


def _get_closest_ak_email_labels(parent_id):
    if not isnan(parent_id):
        parent_email = df[df['id'] == parent_id]
        if parent_email.empty:
            return None, parent_email
        parent_email = parent_email.squeeze()
        parent_email_labels = parent_email['tags']
        if _is_architectural(parent_email_labels):
            return parent_email_labels, parent_email
        parent_id = parent_email['parent_id']
    return None, None


if __name__ == '__main__':
    skip_not_ak = True
    df = pd.read_json('emails_manual.json')
    # df = pd.read_json('emails_bert.json')

    n = 3

    df['tags'] = df['tags'].transform(
        lambda tags: [tag for tag in tags if tag in TAGS]
    )

    labels = TAGS
    labels.remove('not-ak')

    patterns = dict.fromkeys(labels, {})
    for label in labels:
        remaining_labels = [l for l in labels if l != label]
        patterns[label] = dict.fromkeys(remaining_labels, 0)
        if n == 3:
            for tag in remaining_labels:
                patterns[label][tag] = dict.fromkeys([l for l in remaining_labels if l != tag], 0)

    for i in range(len(df)):
        email = df.iloc[i]
        email_labels = email['tags']
        if _is_architectural(email_labels):
            # skip all non-architectural emails
            closest_ak_email_labels, parent_email = _get_closest_ak_email_labels(email['parent_id'])
            if closest_ak_email_labels:
                if n == 3:
                    parent_id = parent_email['parent_id']
                    top_ak_email_labels, _ = _get_closest_ak_email_labels(parent_id)
                    if top_ak_email_labels:
                        for label in email_labels:
                            for closest_label in closest_ak_email_labels:
                                for top_label in top_ak_email_labels:
                                    if label != closest_label and label != top_label and top_label != closest_label:
                                        patterns[top_label][closest_label][label] += 1
                else:
                    for label in email_labels:
                        for closest_label in closest_ak_email_labels:
                            if label != closest_label:
                                patterns[closest_label][label] += 1

    flat_dict = _flatten_dict(patterns)
    sorted_dict = _sort_dict(flat_dict)

    with open('patterns.txt', 'w') as f:
        for pattern, count in sorted_dict.items():
            labels = pattern.split('_')
            f.write(
                f'{labels[0]} $ \\rightarrow $ {labels[1]} $ \\rightarrow $ {labels[2]} & {count} \\\ \hline\n')
    # print('yo')
