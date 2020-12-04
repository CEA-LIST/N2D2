

class UndefinedModelError(RuntimeError):
   def __init__(self, arg):
      super().__init__(arg)

class UndefinedParameterError(RuntimeError):
   def __init__(self, value, obj):
      super().__init__("Parameter \'" + str(value) + "\' does not exist in object of type " + str(type(obj)))