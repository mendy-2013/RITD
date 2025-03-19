import importlib
from collections import OrderedDict

import anyconfig
import munch


class Config(object):
    def __init__(self):
        pass

    def load(self, conf):
        conf = anyconfig.load(conf)     # 加載yaml文件
        return munch.munchify(conf)
        # 生成munch對象,类似于字典,若字典有不存在對象直接報錯不會返回NONE

    def compile(self, conf, return_packages=False):
        packages = conf.get('package', [])
        # 传入的配置 conf 中获取一个名为 'package' 的键的值。如果这个键不存在，它将返回一个空列表 []。这个值被分配给变量 packages。
        defines = {}

        for path in conf.get('import', []):
            parent_conf = self.load(path)
            parent_packages, parent_defines = self.compile(
                parent_conf, return_packages=True)
            # 调用类中的 compile 方法，传递刚刚加载的 parent_conf 作为参数。此外，它设置了 return_packages 参数为 True，以指示 compile
            # 方法返回包含 packages 和 defines 的元组。compile 方法的返回值被拆分成 parent_packages 和 parent_defines 两个变量。
            packages.extend(parent_packages)
            # 将 parent_packages 中的内容扩展（合并）到当前函数中的 packages 列表中
            defines.update(parent_defines)
            # update 方法将 parent_defines 中的内容合并到当前函数中的 defines 字典中

        modules = []
        for package in packages:
            module = importlib.import_module(package)
            # 代码使用 importlib.import_module 函数来动态导入相应的模块。这将根据包名导入相应的 Python 模块，并将其分配给变量 module
            modules.append(module)

        if isinstance(conf['define'], dict):  # 条件语句，它检查配置字典 conf 中的 'define' 键的值是否为字典类型。
            conf['define'] = [conf['define']]
            # 它将字典 conf['define'] 封装在一个包含一个元素的列表中。这样，无论 'define' 键的值最初是单个字典还是一个列表，
            # 它都会确保 'define' 键的值最终是一个包含字典的列表。

        for define in conf['define']:
            name = define.copy().pop('name')
            # 首先创建一个名为 name 的变量。它通过复制 define 字典（以防止修改原始字典）并使用 pop 方法从复制的字典中移除 'name'
            # 键来获取名称。这个名称将用于标识这个定义

            if not isinstance(name, str):
                raise RuntimeError('name must be str')

            defines[name] = self.compile_conf(define, defines, modules)
            # 定义的名称和相应的配置内容传递给 self.compile_conf 方法，然后将结果存储在名为 defines 的字典中。

        if return_packages:
            return packages, defines
        else:
            return defines

    def compile_conf(self, conf, defines, modules):
        if isinstance(conf, (int, float)):
            return conf
        elif isinstance(conf, str):
            if conf.startswith('^'):
                return defines[conf[1:]]
            if conf.startswith('$'):
                return {'class': self.find_class_in_modules(conf[1:], modules)}
            return conf
        #     如果 conf 以 ^ 开头，它将尝试返回 defines 字典中关联的值。具体来说，它会去掉开头的 ^，
        #     然后使用剩下的字符串作为键来查找 defines 字典，并返回对应的值。
        #
        #     如果 conf 以 $ 开头，它将返回一个包含键为 'class' 的字典，
        #     以及与 self.find_class_in_modules(conf[1:], modules) 返回的值相关联。这个部分可能涉及到查找类或模块的操作。
        #
        #     如果以上两个条件都不成立，它将简单地返回 conf 本身。
        elif isinstance(conf, dict):
            if 'class' in conf:
                conf['class'] = self.find_class_in_modules(
                    conf['class'], modules)
            if 'base' in conf:
                base = conf.copy().pop('base')

                if not isinstance(base, str):
                    raise RuntimeError('base must be str')

                conf = {
                    **defines[base],
                    **conf,
                }
            return {key: self.compile_conf(value, defines, modules) for key, value in conf.items()}
        elif isinstance(conf, (list, tuple)):
            return [self.compile_conf(value, defines, modules) for value in conf]
        else:
            return conf

    def find_class_in_modules(self, cls, modules):
        if not isinstance(cls, str):
            raise RuntimeError('class name must be str')

        if cls.find('.') != -1:
            package, cls = cls.rsplit('.', 1)
            module = importlib.import_module(package)
            if hasattr(module, cls):
                return module.__name__ + '.' + cls
        # 如果 cls 中包含句点（.），它将尝试分割类名和包名，并然后使用 importlib.import_module 导入包所对应的模块。
        # 如果成功导入模块并且该模块中存在具有给定类名的属性，它将返回该类的全限定名（模块名 + 类名）。

        for module in modules:
            if hasattr(module, cls):
                return module.__name__ + '.' + cls
        raise RuntimeError('class not found ' + cls)
        # 对于每个模块，使用 hasattr(module, cls) 检查该模块是否包含一个名为 cls 的属性（通常是一个类）。
        #
        # 如果在任何一个模块中找到具有给定类名 cls 的属性，它将返回该类所在模块的名称（module.__name__）和类名，这构成了该类的全限定名。
        #
        # 如果在所有模块中都没有找到具有给定类名 cls 的属性，它会引发一个 RuntimeError 异常，指示没有找到该类


class State:
    def __init__(self, autoload=True, default=None):
        self.autoload = autoload
        self.default = default
        # self.autoload 是一个属性，它的默认值为 True，但可以在创建对象时传入不同的值。这个属性可能表示一个标志，指示状态是否自动加载（autoload）或自动初始化。
        #
        # self.default 是另一个属性，它的默认值为 None，同样可以在创建对象时传入不同的值。这个属性可能表示状态的默认值或初始值


class StateMeta(type):  # StateMeta的元类，它继承自内置的type元类。元类用于控制创建和管理类的行为。
    def __new__(mcs, name, bases, attrs):
        # 这是元类中的特殊方法，用于在创建一个新类时被调用。mcs代表元类本身，name是要创建的类的名称，bases是类的基类，attrs是类的属性字典。
        current_states = []
        for key, value in attrs.items():  # 遍历类的属性字典。
            if isinstance(value, State):  # 检查属性是否是一个State类的实例。
                current_states.append((key, value))

        current_states.sort(key=lambda x: x[0]) # 列表按属性名称进行排序。
        attrs['states'] = OrderedDict(current_states)
        # 将排序后的状态列表转化为有序字典（OrderedDict）并将其存储为类的属性states。
        new_class = super(StateMeta, mcs).__new__(mcs, name, bases, attrs)
        # 使用super调用父类的__new__方法来创建一个新的类。这个新类将包含属性states，表示状态。

        # Walk through the MRO
        states = OrderedDict()  # 创建一个空的有序字典states，用于存储所有继承自基类的状态。
        for base in reversed(new_class.__mro__):
        # 在新类的方法解析顺序（Method Resolution Order, MRO）中反向遍历。

            if hasattr(base, 'states'):
                states.update(base.states)
        new_class.states = states

        for key, value in states.items():
            setattr(new_class, key, value.default)

        return new_class


class Configurable(metaclass=StateMeta):
    def __init__(self, *args, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)

    @staticmethod
    def construct_class_from_config(args):
        cls = Configurable.extract_class_from_args(args)
        return cls(**args)

    @staticmethod
    def extract_class_from_args(args):
        cls = args.copy().pop('class')
        package, cls = cls.rsplit('.', 1)
        module = importlib.import_module(package)
        cls = getattr(module, cls)
        return cls

    def load_all(self, **kwargs):
        for name, state in self.states.items():
            if state.autoload:
                self.load(name, **kwargs)

    def load(self, state_name, **kwargs):
        # FIXME: kwargs should be filtered
        # Args passed from command line
        cmd = kwargs.pop('cmd', dict())
        if state_name in kwargs:
            setattr(self, state_name, self.create_member_from_config(
                (kwargs[state_name], cmd)))
        else:
            setattr(self, state_name, self.states[state_name].default)

    def create_member_from_config(self, conf):
        args, cmd = conf
        if args is None or isinstance(args, (int, float, str)):
            return args
        elif isinstance(args, (list, tuple)):
            return [self.create_member_from_config((subargs, cmd)) for subargs in args]
        elif isinstance(args, dict):
            if 'class' in args:
                cls = self.extract_class_from_args(args)
                return cls(**args, cmd=cmd)
            return {key: self.create_member_from_config((subargs, cmd)) for key, subargs in args.items()}
        else:
            return args

    def dump(self):
        state = {}
        state['class'] = self.__class__.__module__ + \
            '.' + self.__class__.__name__
        for name, value in self.states.items():
            obj = getattr(self, name)
            state[name] = self.dump_obj(obj)
        return state

    def dump_obj(self, obj):
        if obj is None:
            return None
        elif hasattr(obj, 'dump'):
            return obj.dump()
        elif isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.dump_obj(value) for value in obj]
        elif isinstance(obj, dict):
            return {key: self.dump_obj(value) for key, value in obj.items()}
        else:
            return str(obj)

