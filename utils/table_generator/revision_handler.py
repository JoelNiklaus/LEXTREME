import json as js
from typing import Union
from helper import TABLE_TO_REPORT_SPECS


class RevisionHandler:

    def __init__(self, report_spec: dict) -> None:
        self.report_specs = self.read_report_specs(report_spec=report_spec)
        self.revision_lookup_table = self.prepare_revision_lookup_table()
        if self.report_specs:
            self.model_order_acc_to_report_specs = [_name_or_path.split('@')[0] for _name_or_path in
                                                    self.report_specs['_name_or_path']]

    def read_report_specs(self, report_spec: str) -> dict:
        with open(TABLE_TO_REPORT_SPECS + f"/{report_spec}.json", 'r') as fname:
            report_spec = js.load(fname)
        return report_spec

    def prepare_revision_lookup_table(self) -> dict:

        revision_lookup_table = dict()
        for name_or_path_and_revision in self.report_specs['_name_or_path']:
            _name_or_path, revision = name_or_path_and_revision.split('@')
            revision_lookup_table[_name_or_path] = revision
        return revision_lookup_table

    def insert_revision(self, _name_or_path: str) -> str:
        if _name_or_path in self.revision_lookup_table.keys():
            return self.revision_lookup_table[_name_or_path]
        else:
            'no information'

    def revision_does_match(self, _name_or_path: str, revision_from_wandb: str) -> Union[bool, str]:
        if _name_or_path in self.revision_lookup_table.keys():
            if revision_from_wandb == self.revision_lookup_table[_name_or_path]:
                return True
            else:
                return False
        else:
            return 'no information'
