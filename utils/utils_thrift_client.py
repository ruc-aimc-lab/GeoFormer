import euler
import thriftpy2
from euler import base_compat_middleware


class ThriftDalBase(object):
    def __init__(self, client, instance):
        self._client = client
        self._instance = instance
        mod_attrs = filter(lambda x: x[0: 2] != '__', instance.__dict__.keys())
        self._mod_attrs = {k: getattr(instance, k) for k in mod_attrs}

    def __getattr__(self, attr):
        if attr in self._mod_attrs:
            return self._mod_attrs[attr]
        if attr[0: 2] == '_m' and self._client:
            return getattr(self._client, attr[2:])
        if attr[0: 2] == '_r':
            return getattr(self._instance, attr[2:])
        if self._client:
            return getattr(self._client, attr)
        else:
            raise Exception("no such attr")


def get_thrift(full_idl, psm, cluster='default', idc=None, service=None, only_module=False, **kwargs):
    module = full_idl.split('/')[-1].replace('.', '_')
    if not service and not only_module:
        slist = []
        f = open(full_idl, 'r')
        for line in f:
            line = line.strip('\t {\r\n')
            if line.find('service ') != 0:
                continue
            line = line[len('service '):]
            if line:
                slist.append(line)
        f.close()
        if len(slist) != 1:
            raise Exception('cannot determine a service for idl ' + full_idl)
        service = slist[0]
    instance = thriftpy2.load(full_idl, module_name=module)
    if only_module:
        client = None
    else:
        consul_name = 'sd://%s?cluster=%s' % (psm, cluster)
        if idc:
            consul_name = consul_name + '&idc=' + idc
        client = euler.Client(getattr(instance, service), consul_name, **kwargs)
        client.use(base_compat_middleware.client_middleware)
    typeo = type(module + '_dal', (ThriftDalBase, ), {})
    return typeo(client, instance)


def get_local_thrift(full_idl, consul, service, **kwargs):
    module = full_idl.split('/')[-1].replace('.', '_')
    instance = thriftpy2.load(full_idl, module_name=module)
    client = euler.Client(getattr(instance, service), consul, **kwargs)
    client.use(base_compat_middleware.client_middleware)
    typeo = type(module + '_dal', (ThriftDalBase, ), {})
    return typeo(client, instance)