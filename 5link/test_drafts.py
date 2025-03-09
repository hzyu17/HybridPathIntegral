import jax.numpy as jnp
from rabbit_kinematics import *
from rabbit_dynamics import *

def p_Torso(var1):
    """
    Compute the torso position vector p_Torso from var1.
    
    Parameters:
      var1 : a JAX array of shape (7,) representing the input vector.
             (In the original MATLAB code, var1 is a 7x1 vector.)
             
    Returns:
      A JAX array of shape (3,) representing the output:
          [ var1[0] + 0.63*sin(var1[2]),
            0,
            0.63*cos(var1[2]) + var1[1] ]
    """
    # Ensure var1 is a flat vector of length 7.
    var1 = jnp.ravel(var1)
    if var1.shape[0] != 7:
        raise ValueError("Input var1 must be a vector of length 7")
    
    p0 = var1[0] + 0.63 * jnp.sin(var1[2])
    p1 = 0.0
    p2 = 0.63 * jnp.cos(var1[2]) + var1[1]
    
    return jnp.array([p0, p1, p2])


def p_RightToe(var1):
    """
    Compute p_RightToe from var1.
    
    Parameters:
      var1 : a JAX array of shape (7,) representing the input.
      
    Returns:
      A JAX array of shape (3,) computed as:
      
        output[0] = 0 + t1541*t1555 + 0.8*(t1523*t1555 + t1563*t1564) + t1563*t1574 + var1[0]
        output[1] = 0
        output[2] = 0 + t1541*t1563 + t1574*t1595 + 0.8*(t1523*t1563 + t1564*t1595) + var1[1]
      
      where the temporary variables are computed as follows:
      
        t1549 = cos(var1[2])
        t1543 = cos(var1[3])
        t1544 = sin(var1[2])
        t1551 = sin(var1[3])
        t1523 = cos(var1[4])
        t1547 = t1543 * t1544
        t1554 = t1549 * t1551
        t1555 = t1547 + t1554
        t1561 = t1549 * t1543
        t1562 = - t1544 * t1551
        t1563 = t1561 + t1562
        t1564 = sin(var1[4])
        t1526 = - t1523
        t1534 = 1. + t1526
        t1540 = 0.4 * t1534
        t1541 = 0. + t1540
        t1567 = -0.4 * t1564
        t1574 = 0. + t1567
        t1590 = - t1543 * t1544
        t1594 = - t1549 * t1551
        t1595 = t1590 + t1594
    """
    # Ensure var1 is a flat vector of length 7.
    var1 = jnp.ravel(var1)
    if var1.shape[0] != 7:
        raise ValueError("Input var1 must be a vector of length 7")
    
    t1549 = jnp.cos(var1[2])
    t1543 = jnp.cos(var1[3])
    t1544 = jnp.sin(var1[2])
    t1551 = jnp.sin(var1[3])
    t1523 = jnp.cos(var1[4])
    t1547 = t1543 * t1544
    t1554 = t1549 * t1551
    t1555 = t1547 + t1554
    t1561 = t1549 * t1543
    t1562 = - t1544 * t1551
    t1563 = t1561 + t1562
    t1564 = jnp.sin(var1[4])
    t1526 = - t1523
    t1534 = 1. + t1526
    t1540 = 0.4 * t1534
    t1541 = 0. + t1540
    t1567 = -0.4 * t1564
    t1574 = 0. + t1567
    t1590 = - t1543 * t1544
    t1594 = - t1549 * t1551
    t1595 = t1590 + t1594

    p0 = t1541 * t1555 + 0.8 * (t1523 * t1555 + t1563 * t1564) + t1563 * t1574 + var1[0]
    p1 = 0.0
    p2 = t1541 * t1563 + t1574 * t1595 + 0.8 * (t1523 * t1563 + t1564 * t1595) + var1[1]
    
    return jnp.array([p0, p1, p2])


def p_q2_right(var1):
    """
    Compute p_q2_right from var1.
    
    Input:
      var1: a JAX array of shape (7,). (In MATLAB, var1 is 7x1.)
      
    Returns:
      A JAX array of shape (3,) computed as follows:
      
         t1517 = cos(var1[2])
         t1511 = cos(var1[3])
         t1512 = sin(var1[2])
         t1522 = sin(var1[3])
         t1490 = cos(var1[4])
         t1514 = t1511 * t1512
         t1523 = t1517 * t1522
         t1526 = t1514 + t1523
         t1528 = t1517 * t1511
         t1530 = - t1512 * t1522
         t1534 = t1528 + t1530
         t1535 = sin(var1[4])
         t1493 = - t1490
         t1508 = 1. + t1493
         t1509 = 0.4 * t1508
         t1510 = t1509
         t1537 = -0.4 * t1535
         t1540 = t1537
         t1552 = - t1511 * t1512
         t1553 = - t1517 * t1522
         t1554 = t1552 + t1553
         
         out[0] = t1510*t1526 + 0.4*(t1490*t1526 + t1534*t1535) + t1534*t1540 + var1[0]
         out[1] = 0
         out[2] = t1510*t1534 + t1540*t1554 + 0.4*(t1490*t1534 + t1535*t1554) + var1[1]
    """
    # Flatten and check input shape.
    var1 = jnp.ravel(var1)
    if var1.shape[0] != 7:
        raise ValueError("Input var1 must be a vector of length 7.")
    
    t1517 = jnp.cos(var1[2])
    t1511 = jnp.cos(var1[3])
    t1512 = jnp.sin(var1[2])
    t1522 = jnp.sin(var1[3])
    t1490 = jnp.cos(var1[4])
    t1514 = t1511 * t1512
    t1523 = t1517 * t1522
    t1526 = t1514 + t1523
    t1528 = t1517 * t1511
    t1530 = - t1512 * t1522
    t1534 = t1528 + t1530
    t1535 = jnp.sin(var1[4])
    t1493 = - t1490
    t1508 = 1. + t1493
    t1509 = 0.4 * t1508
    t1510 = t1509
    t1537 = -0.4 * t1535
    t1540 = t1537
    t1552 = - t1511 * t1512
    t1553 = - t1517 * t1522
    t1554 = t1552 + t1553

    out0 = t1510 * t1526 + 0.4 * (t1490 * t1526 + t1534 * t1535) + t1534 * t1540 + var1[0]
    out1 = 0.0
    out2 = t1510 * t1534 + t1540 * t1554 + 0.4 * (t1490 * t1534 + t1535 * t1554) + var1[1]
    
    return jnp.array([out0, out1, out2])


def p_q2_left(var1):
    """
    Compute p_q2_left from var1.
    
    Input:
      var1: a JAX array of shape (7,), where:
            var1[0] = first element, ..., var1[6] = seventh element.
    
    Returns:
      A JAX array of shape (3,) computed as:
      
          out[0] = t1526*t1541 + 0.4*(t1509*t1541 + t1547*t1549) + t1547*t1554 + var1[0]
          out[1] = 0
          out[2] = t1526*t1547 + t1554*t1574 + 0.4*(t1509*t1547 + t1549*t1574) + var1[1]
    
    The intermediate variables are computed as:
      t1535 = cos(var1[2])
      t1527 = cos(var1[5])
      t1528 = sin(var1[2])
      t1537 = sin(var1[5])
      t1509 = cos(var1[6])
      t1534 = t1527 * t1528
      t1540 = t1535 * t1537
      t1541 = t1534 + t1540
      t1544 = t1535 * t1527
      t1545 = - t1528 * t1537
      t1547 = t1544 + t1545
      t1549 = sin(var1[6])
      t1510 = - t1509
      t1514 = 1. + t1510
      t1523 = 0.4 * t1514
      t1526 = t1523   # (0. + t1523)
      t1551 = -0.4 * t1549
      t1554 = t1551   # (0. + t1551)
      t1569 = - t1527 * t1528
      t1570 = - t1535 * t1537
      t1574 = t1569 + t1570
    """
    # Ensure var1 is a flat vector of length 7.
    var1 = jnp.ravel(var1)
    if var1.shape[0] != 7:
        raise ValueError("Input var1 must be a vector of length 7")
    
    t1535 = jnp.cos(var1[2])
    t1527 = jnp.cos(var1[5])
    t1528 = jnp.sin(var1[2])
    t1537 = jnp.sin(var1[5])
    t1509 = jnp.cos(var1[6])
    t1534 = t1527 * t1528
    t1540 = t1535 * t1537
    t1541 = t1534 + t1540
    t1544 = t1535 * t1527
    t1545 = - t1528 * t1537
    t1547 = t1544 + t1545
    t1549 = jnp.sin(var1[6])
    t1510 = - t1509
    t1514 = 1. + t1510
    t1523 = 0.4 * t1514
    t1526 = t1523
    t1551 = -0.4 * t1549
    t1554 = t1551
    t1569 = - t1527 * t1528
    t1570 = - t1535 * t1537
    t1574 = t1569 + t1570

    out0 = t1526 * t1541 + 0.4 * (t1509 * t1541 + t1547 * t1549) + t1547 * t1554 + var1[0]
    out1 = 0.0
    out2 = t1526 * t1547 + t1554 * t1574 + 0.4 * (t1509 * t1547 + t1549 * t1574) + var1[1]
    
    return jnp.array([out0, out1, out2])


def p_LeftToe(var1):
    """
    Compute p_LeftToe from var1.
    
    Parameters:
      var1: a JAX array of shape (7,) (or convertible to a 7-element vector).
            (Assumes angles are in radians.)
    
    Returns:
      A JAX array of shape (3,) computed as:
      
          p[0] = t1555*t1579 + 0.8*(t1540*t1579 + t1585*t1588) + t1585*t1595 + var1[0]
          p[1] = 0
          p[2] = t1555*t1585 + t1595*t1609 + 0.8*(t1540*t1585 + t1588*t1609) + var1[1]
      
      with the intermediate variables defined as follows:
      
          t1564 = cos(var1[2])
          t1556 = cos(var1[5])
          t1561 = sin(var1[2])
          t1567 = sin(var1[5])
          t1540 = cos(var1[6])
          t1563 = t1556 * t1561
          t1574 = t1564 * t1567
          t1579 = t1563 + t1574
          t1583 = t1564 * t1556
          t1584 = - t1561 * t1567
          t1585 = t1583 + t1584
          t1588 = sin(var1[6])
          t1541 = - t1540
          t1547 = 1. + t1541         # equals 1 - cos(var1[6])
          t1554 = 0.4 * t1547
          t1555 = t1554
          t1589 = -0.4 * t1588
          t1595 = t1589
          t1604 = - t1556 * t1561
          t1605 = - t1564 * t1567
          t1609 = t1604 + t1605   # = - (t1556*t1561 + t1564*t1567)
    """
    # Ensure var1 is a flat vector of length 7.
    var1 = jnp.ravel(var1)
    if var1.shape[0] != 7:
        raise ValueError("Input var1 must be a vector of length 7")
    
    t1564 = jnp.cos(var1[2])
    t1556 = jnp.cos(var1[5])
    t1561 = jnp.sin(var1[2])
    t1567 = jnp.sin(var1[5])
    t1540 = jnp.cos(var1[6])
    t1563 = t1556 * t1561
    t1574 = t1564 * t1567
    t1579 = t1563 + t1574
    t1583 = t1564 * t1556
    t1584 = - t1561 * t1567
    t1585 = t1583 + t1584
    t1588 = jnp.sin(var1[6])
    t1541 = - t1540
    t1547 = 1. + t1541
    t1554 = 0.4 * t1547
    t1555 = t1554
    t1589 = -0.4 * t1588
    t1595 = t1589
    t1604 = - t1556 * t1561
    t1605 = - t1564 * t1567
    t1609 = t1604 + t1605

    p0 = t1555 * t1579 + 0.8 * (t1540 * t1579 + t1585 * t1588) + t1585 * t1595 + var1[0]
    p1 = 0.0
    p2 = t1555 * t1585 + t1595 * t1609 + 0.8 * (t1540 * t1585 + t1588 * t1609) + var1[1]
    
    return jnp.array([p0, p1, p2])


# Example usage:
if __name__ == '__main__':
    # Create an example input vector of length 7.
    q = jnp.array([1.0, 2.0, 0.5, 0.3, 0.2, 0.1, 0.0])
    dotq = jnp.zeros(7)
    u = jnp.zeros(4)
    x_init = jnp.concatenate([q, dotq])
    result = fxgu_floating_base(x_init, u)
    print("fxgu_floating_base output:", result)