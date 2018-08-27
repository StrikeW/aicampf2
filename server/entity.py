
class ModelDO(object):
    def __init__(self):
        self.mid = 0
        self.type = 1
        self.saved_path = 'saved path'


    def add_flavor(self, name, **params):
        """Add an entry for how to serve the model in a given format."""
        self.flavors[name] = params
        return self

    def save(self, path):
        """Write this model as a YAML file to a local file."""
        with open(path, 'w') as out:
            self.to_yaml(out)

    def predict(self):
        pass

    @staticmethod
    def get_by_id(mid):
        model = ModelDO()
        model.mid = 1
        model.saved_path = 'saved_model'
        return model


