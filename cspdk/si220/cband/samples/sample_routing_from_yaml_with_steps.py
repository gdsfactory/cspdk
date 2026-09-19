"""Sample pads with routes."""

if __name__ == "__main__":
    import gdsfactory as gf

    from cspdk.si220.cband import PDK

    PDK.activate()

    yaml = """
# yaml-language-server: $schema=../build/schemas/top.json
instances:
  tr:
    component: pad
    settings: {}
  bl:
    component: pad
    settings: {}
  br:
    component: pad
    settings: {}
  obstacle:
    component: pad
    settings: {}
  tl:
    component: pad
    settings: {}
connections: {}
routes:
  bundle:
    links:
      tl,e3: tr,e1
      bl,e3: br,e1
    routing_strategy: route_bundle_metal
    settings:
      min_straight_taper: 3
      separation: 10
      layer_marker: [1, 0]
      steps:
      - dx: 400.0
        dy: 20
      - dy: 100.0
nets: []
ports: {}
placements:
  tr:
    x: 0.0
    y: 0.0
    dx: 1203.52099609375
    dy: 808.427001953125
    rotation: 0.0
    mirror: false
  bl:
    x: -97.469
    y: -22.356
    dx: 542.6489868164062
    dy: 177.4250030517578
    rotation: 0.0
    mirror: false
  br:
    x: 0.0
    y: 0.0
    dx: 1194.5660400390625
    dy: 597.3690185546875
    rotation: 0.0
    mirror: false
  obstacle:
    x: 494.896
    y: 157.058
    dx: 129.258
    dy: 288.459
  tl:
    x: 0.0
    y: 0.0
    dx: 356.6059875488281
    dy: 313.4670104980469
    rotation: 0.0
    mirror: false


"""

    c = gf.read.from_yaml(yaml)

    c.show()
