from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def connect_to_coppeliasim():
    '''Connect to CoppeliaSim through the ZMQ remote API server.'''
    client = RemoteAPIClient()

    #Access the remote 'sim' Object:
    sim = client.require('sim')
    print("Connected to CoppeliaSim!")
    return sim