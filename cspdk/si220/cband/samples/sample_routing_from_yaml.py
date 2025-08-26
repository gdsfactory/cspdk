"""Sample pads with routes."""

if __name__ == "__main__":
    import gdsfactory as gf

    from cspdk.si220.cband import PDK

    PDK.activate()

    yaml = """

# yaml-language-server: $schema=../build/schemas/top.json
instances:
  instance4:
    component: pad
    settings: {}
  instance1:
    component: pad
    settings: {}
  instance3:
    component: pad
    settings: {}
  instance2:
    component: pad
    settings: {}
connections: {}
routes:
  bundle:
    links:
      instance1-3,e3: instance2-4,e1
    routing_strategy: route_bundle_metal
    settings:
      end_straight_length: 3
      min_straight_taper: 3
      separation: 3
      sort_ports: true
      start_straight_length: 3
nets: []
ports: {}
placements:
  instance4:
    x: 0.0
    y: 0.0
    dx: 931.0809936523438
    dy: 376.8970031738281
    rotation: 0.0
    mirror: false
  instance1:
    x: 0.0
    y: 0.0
    dx: -120.66300201416016
    dy: 216.0260009765625
    rotation: 0.0
    mirror: false
  instance3:
    x: -97.469
    y: -22.356
    dx: -14.163
    dy: -98.992
  instance2:
    x: 0.0
    y: 0.0
    dx: 884.3419799804688
    dy: 804.1840209960938
    rotation: 0.0
    mirror: false

"""

    c = gf.read.from_yaml(yaml)

    c.show()
