from math import isnan

import pandas
import pandas as pd

from dl.constants import TAGS


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
    return label != [] and label != [''] and 'not-ak' not in label


def _get_closest_ak_email_labels(df: pandas.DataFrame, parent_id, skip_not_ak):
    """
    Return the parent email and its labels. If skip_not_ak is True, returns the
    closest architectural email.
    """
    if not isnan(parent_id):
        parent_email = df[df['ID'] == parent_id]
        if parent_email.empty:
            return None, parent_email
        parent_email = parent_email.squeeze()
        parent_email_labels = parent_email['TAGS']
        if _is_architectural(parent_email_labels):
            return parent_email_labels, parent_email
        parent_id = parent_email['PARENT_ID']

        if skip_not_ak:
            return _get_closest_ak_email_labels(df, parent_id, skip_not_ak)
        return None, parent_email
    return None, None


def generate_patterns(df: pandas.DataFrame, dataset: str = 'manual', skip_not_ak=False):
    n = 2
    labels = TAGS[:4]

    patterns = dict.fromkeys(labels, {})
    for label in labels:
        remaining_labels = [l for l in labels if l != label]
        patterns[label] = dict.fromkeys(remaining_labels, 0)
        if n == 3:
            for tag in remaining_labels:
                patterns[label][tag] = dict.fromkeys([l for l in remaining_labels if l != tag], 0)

    for i in range(len(df)):
        email = df.iloc[i]
        email_labels = email['TAGS']
        if _is_architectural(email_labels):
            # skip all non-architectural emails
            closest_ak_email_labels, parent_email = _get_closest_ak_email_labels(df, email['PARENT_ID'], skip_not_ak)
            if closest_ak_email_labels:
                if n == 3:
                    parent_id = parent_email['PARENT_ID']
                    top_ak_email_labels, _ = _get_closest_ak_email_labels(parent_id, skip_not_ak)
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

    skip = '_skip' if skip_not_ak else ''

    with open(f'analysis/patterns_{dataset}{skip}.txt', 'w') as f:
        for pattern, count in sorted_dict.items():
            labels = pattern.split('_')
            f.write(
                f'{labels[0]} $ \\rightarrow $ {labels[1]} & {count} \\\ \hline\n')


if __name__ == '__main__':
    df_manual = pd.read_json('data/labeled_dataset_preprocessed.json', orient='index')
    df_bert = pd.read_json('data/bert_classified_emails.json', orient='index')

    # Transform labels to lists
    df_manual['TAGS'] = df_manual['TAGS'].transform(
        lambda tags: tags.split(', ')
    )
    df_bert['TAGS'] = df_bert['TAGS'].transform(
        lambda tags: tags.split(', ')
    )

    generate_patterns(df_manual)
    generate_patterns(df_bert, 'bert')

    generate_patterns(df_manual, skip_not_ak=True)
    generate_patterns(df_bert, 'bert', skip_not_ak=True)
