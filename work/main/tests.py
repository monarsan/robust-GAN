from gan import gan as GAN
import numpy as np

z = np.array([
    [1, 2, 3],
    [4, 5, 6]]
)
par_a = np.array(
    [2, 3, 4]
)
par_b = np.array(
    [-1, 0, 1]
)
z_sq = z ** 2


def test_discriminator_in_matrixpy():
    '''
    Test this code : t0_z = z_sq@par_a + z@par_b - bias in matrix.py
    '''
    obj = z_sq @ par_a
    # operator @
    assert obj[0] == np.array([
        2 + 12 + 36
    ])
    assert obj[1] == np.array([
        32 + 75 + 144
    ])
    obj = z @ par_b
    # operator @
    assert obj[0] == np.array([
        -1 + 3
    ])
    assert obj[1] == np.array([
        -4 + 6
    ])
    # bias
    obj = z_sq @ par_a - 1
    assert obj[0] == np.array([
        2 + 12 + 36 - 1
    ])
    assert obj[1] == np.array([
        32 + 75 + 144 - 1
    ])


def test_discriminator_in_ganpy():
    gan = GAN(3, 0.1)
    gan.setting = 'mu'
    gan.D = np.concatenate([par_a, par_b])
    assert list(gan.D) == [2, 3, 4, -1, 0, 1]
    obj = gan._u(z)
    assert list(np.sum((z ** 2) * gan.D[:gan.data_dim], axis=1))\
        == [2 + 12 + 36, 32 + 75 + 144]
    assert obj[0] == np.array([
        2 + 12 + 36 + 2
    ])
    assert obj[1] == np.array([
        32 + 75 + 144 + 2
    ])
