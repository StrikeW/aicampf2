
# coding: utf-8
from sqlalchemy import Column, Float, ForeignKey, Integer, String, TIMESTAMP, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Model(Base):
    __tablename__ = 'model'

    mid = Column(Integer, primary_key=True)
    type = Column(Integer, nullable=False, index=True)
    name = Column(String(64))
    saved_path = Column(String(255))
    hyper_params = Column(String(255))
    accuracy = Column(Float(asdecimal=True))
    rmse = Column(Float(asdecimal=True))
    train_time = Column(Integer, nullable=False, server_default=text("'0'"))
    ctime = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


class Deploy(Base):
    __tablename__ = 'deploy'

    id = Column(Integer, primary_key=True)
    mid = Column(ForeignKey('model.mid'), nullable=False, index=True)
    state = Column(Integer)
    deploy_time = Column(Integer, server_default=text("'0'"))
    ctime = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    model = relationship('Model')


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


