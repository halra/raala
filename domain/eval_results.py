import json

class EvalResults:
    def __init__(self, created_at, labels, true_labels=None, predicted_labels=None, instance_name="foo"):
        self.created_at = created_at
        self.labels = labels
        self.instance_name = instance_name
        self.seeds = []
        self.epochs = []
        self.model_names = []
        self.true_labels = true_labels if true_labels is not None else []
        self.predicted_labels = predicted_labels if predicted_labels is not None else []

    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        instance = cls(
            created_at=data['created_at'],
            labels=data['labels'],
            instance_name=data.get('instance_name', "foo"),
            true_labels=data.get('true_labels', []),
            predicted_labels=data.get('predicted_labels', [])
        )
        instance.seeds = data.get('seeds', [])
        instance.epochs = data.get('epochs', [])
        instance.model_names = data.get('model_names', [])
        return instance
