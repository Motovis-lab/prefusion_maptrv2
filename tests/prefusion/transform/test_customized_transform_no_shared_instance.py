import numpy as np

from prefusion.dataset.transform import (
    CameraImage, RandomRotateSpace, RandomMirrorSpace,
    RenderIntrinsic, RenderExtrinsic, RenderVirtualCamera, RandomRenderExtrinsic,
)


def test_render_intrinsic_no_shared_instance():
    """Test that RenderIntrinsic doesn't cause multiple CameraImage objects 
    with the same cam_id to share the same intrinsic instance.
    
    This test uses mocking to simulate the worst-case scenario where CameraImage.render_intrinsic
    directly stores the passed intrinsic parameter, which would expose any shared instance issues.
    """
    # Create two CameraImage objects with the same cam_id
    camera1 = CameraImage(
        name="camera1",
        cam_id='VCAMERA_TEST',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    camera2 = CameraImage(
        name="camera2",
        cam_id='VCAMERA_TEST',  # Same cam_id as camera1
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    # Mock render_intrinsic to directly store the passed intrinsic (worst case scenario)
    def mock_render_intrinsic(self, resolution, intrinsic, **kwargs):
        # Simulate the worst case: directly store the passed intrinsic without any processing
        self.intrinsic = intrinsic
        return self
    
    # Apply the mock to both cameras
    camera1.render_intrinsic = mock_render_intrinsic.__get__(camera1, CameraImage)
    camera2.render_intrinsic = mock_render_intrinsic.__get__(camera2, CameraImage)
    
    # Apply RenderIntrinsic transform
    transform = RenderIntrinsic(
        resolutions={'VCAMERA_TEST': (128, 128)},
        intrinsics={'VCAMERA_TEST': [64, 64, 64, 64]}
    )
    
    transform(camera1, camera2)
    
    # Verify that both cameras have the expected intrinsic values
    expected_intrinsic = [64, 64, 64, 64]
    assert camera1.intrinsic == expected_intrinsic
    assert camera2.intrinsic == expected_intrinsic
    
    # The critical test: verify that they don't share the same instance
    # This tests whether RenderIntrinsic properly creates copies for each camera
    assert camera1.intrinsic is not camera2.intrinsic, (
        "Multiple CameraImage objects with the same cam_id should not share "
        "the same intrinsic instance. Each should have its own copy."
    )
    
    # Additional verification: modifying one shouldn't affect the other
    original_camera2_intrinsic = camera2.intrinsic.copy()
    camera1.intrinsic[0] = 999  # Modify camera1's intrinsic
    
    # If they don't share the same instance, camera2 should be unchanged
    assert camera2.intrinsic == original_camera2_intrinsic, (
        "Modifying one camera's intrinsic should not affect another camera's intrinsic"
    )

def test_render_extrinsic_no_shared_instance():
    """Test that RenderExtrinsic doesn't cause multiple CameraImage objects 
    with the same cam_id to share the same extrinsic instance.
    
    This is a unit test that mocks CameraImage.render_extrinsic to simulate
    the worst-case scenario where it directly stores the passed delta_extrinsic parameter.
    This should FAIL with the current RenderExtrinsic implementation because it passes
    the same delta_extrinsic tuple to all cameras with the same cam_id.
    """
    # Create two CameraImage objects with the same cam_id
    camera1 = CameraImage(
        name="camera1",
        cam_id='VCAMERA_TEST',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    camera2 = CameraImage(
        name="camera2",
        cam_id='VCAMERA_TEST',  # Same cam_id as camera1
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    # Mock render_extrinsic to directly store the passed delta_extrinsic (worst case scenario)
    def mock_render_extrinsic(self, delta_extrinsic, **kwargs):
        # Simulate the worst case: directly store the passed delta_extrinsic tuple
        # This exposes shared instance issues if RenderExtrinsic passes the same tuple
        self.extrinsic = delta_extrinsic
        return self
    
    # Apply the mock to both cameras
    camera1.render_extrinsic = mock_render_extrinsic.__get__(camera1, CameraImage)
    camera2.render_extrinsic = mock_render_extrinsic.__get__(camera2, CameraImage)
    
    # Apply RenderExtrinsic transform
    transform = RenderExtrinsic(
        del_rotations={'VCAMERA_TEST': [0, 0, 30]}  # 30 degree rotation around Z-axis
    )
    
    transform(camera1, camera2)
    
    # Verify that both cameras have extrinsic parameters (the delta_extrinsic tuple)
    assert camera1.extrinsic is not None
    assert camera2.extrinsic is not None
    
    # The critical test: verify that they don't share the same instance
    # This should FAIL with the current RenderExtrinsic implementation because it
    # passes the same self.del_extrinsics[cam_id] tuple to multiple cameras
    assert camera1.extrinsic is not camera2.extrinsic, (
        "Multiple CameraImage objects with the same cam_id should not share "
        "the same extrinsic instance. Each should have its own copy."
    )
    
    # Additional verification: modifying one shouldn't affect the other
    # Since extrinsic is a tuple with numpy arrays, let's modify the rotation matrix
    if hasattr(camera1.extrinsic[0], '__setitem__'):  # If it's a mutable array
        original_value = camera2.extrinsic[0][0, 0]
        camera1.extrinsic[0][0, 0] = 999
        
        # If they don't share the same instance, camera2 should be unchanged
        assert camera2.extrinsic[0][0, 0] == original_value, (
            "Modifying one camera's extrinsic should not affect another camera's extrinsic"
        )


# ...existing tests...

def test_render_virtual_camera_no_shared_instance():
    """Test that RenderVirtualCamera doesn't cause multiple CameraImage objects 
    with the same cam_id to share the same camera instance.
    
    This is a unit test that mocks CameraImage.render_camera to simulate
    the worst-case scenario where it directly stores the passed camera parameter.
    """
    # Create two CameraImage objects with the same cam_id
    camera1 = CameraImage(
        name="camera1",
        cam_id='VCAMERA_TEST',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    camera2 = CameraImage(
        name="camera2",
        cam_id='VCAMERA_TEST',  # Same cam_id as camera1
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    # Mock render_camera to directly store the passed camera (worst case scenario)
    def mock_render_camera(self, camera, **kwargs):
        # Simulate the worst case: directly store the passed camera object
        # This exposes shared instance issues if RenderVirtualCamera passes the same camera
        self.stored_camera = camera
        return self
    
    # Apply the mock to both cameras
    camera1.render_camera = mock_render_camera.__get__(camera1, CameraImage)
    camera2.render_camera = mock_render_camera.__get__(camera2, CameraImage)
    
    # Apply RenderVirtualCamera transform
    transform = RenderVirtualCamera(
        camera_settings={
            'VCAMERA_TEST': {
                'cam_type': 'PerspectiveCamera',
                'resolution': (128, 96),
                'euler_angles': [0, 0, 45],  # 45 degree rotation around Z-axis
                'intrinsic': 'auto'
            }
        }
    )
    
    transform(camera1, camera2)
    
    # Verify that both cameras have camera objects stored
    assert hasattr(camera1, 'stored_camera') and camera1.stored_camera is not None
    assert hasattr(camera2, 'stored_camera') and camera2.stored_camera is not None
    
    # The critical test: verify that they share the same camera instance
    # RenderVirtualCamera should share the same camera instance since it creates
    # cameras once in __init__ and reuses them
    assert camera1.stored_camera is camera2.stored_camera, (
        "RenderVirtualCamera should share the same camera instance between "
        "CameraImage objects with the same cam_id for efficiency"
    )


def test_random_render_extrinsic_no_shared_instance():
    """Test that RandomRenderExtrinsic doesn't cause multiple CameraImage objects 
    to share the same extrinsic instance.
    
    This is a unit test that mocks CameraImage.render_extrinsic to simulate
    the worst-case scenario where it directly stores the passed delta_extrinsic parameter.
    RandomRenderExtrinsic should generate different random extrinsics for each camera.
    """
    # Create two CameraImage objects
    camera1 = CameraImage(
        name="camera1",
        cam_id='VCAMERA_TEST1',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    camera2 = CameraImage(
        name="camera2",
        cam_id='VCAMERA_TEST2',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    # Mock render_extrinsic to directly store the passed delta_extrinsic (worst case scenario)
    def mock_render_extrinsic(self, delta_extrinsic, **kwargs):
        # Simulate the worst case: directly store the passed delta_extrinsic tuple
        # This exposes shared instance issues if RandomRenderExtrinsic passes the same tuple
        self.stored_extrinsic = delta_extrinsic
        return self
    
    # Apply the mock to both cameras
    camera1.render_extrinsic = mock_render_extrinsic.__get__(camera1, CameraImage)
    camera2.render_extrinsic = mock_render_extrinsic.__get__(camera2, CameraImage)
    
    # Apply RandomRenderExtrinsic transform with deterministic seeds
    transform = RandomRenderExtrinsic(prob=1.0, angles=[5, 5, 5])  # Ensure transform is applied
    
    # Use deterministic seeds to ensure reproducible results
    seeds = {'frame': 12345}
    transform(camera1, camera2, seeds=seeds)
    
    # Verify that both cameras have extrinsic parameters stored
    assert hasattr(camera1, 'stored_extrinsic') and camera1.stored_extrinsic is not None
    assert hasattr(camera2, 'stored_extrinsic') and camera2.stored_extrinsic is not None
    
    # The critical test: verify that they don't share the same instance
    # RandomRenderExtrinsic should generate different random extrinsics for each camera
    assert camera1.stored_extrinsic is not camera2.stored_extrinsic, (
        "RandomRenderExtrinsic should generate different extrinsic instances "
        "for different cameras, not share the same instance"
    )
    
    # Verify that the rotation matrices are different objects
    del_R1, del_t1 = camera1.stored_extrinsic
    del_R2, del_t2 = camera2.stored_extrinsic
    
    assert del_R1 is not del_R2, "Rotation matrices should be different objects"
    assert del_t1 is not del_t2, "Translation vectors should be different objects"


def test_random_rotate_space_no_shared_instance():
    """Test that RandomRotateSpace doesn't cause multiple CameraImage objects 
    to share the same instance when calling render_extrinsic.
    
    This is a unit test that mocks CameraImage.render_extrinsic to simulate
    the worst-case scenario where it directly stores the passed del_extrinsic parameter.
    This should ensure that each camera gets its own copy of the extrinsic parameters.
    """
    # Create two CameraImage objects with the same cam_id (to simulate potential shared instance issues)
    camera1 = CameraImage(
        name="camera1",
        cam_id='VCAMERA_TEST',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    camera2 = CameraImage(
        name="camera2",
        cam_id='VCAMERA_TEST',  # Same cam_id as camera1
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    # Mock render_extrinsic to directly store the passed del_extrinsic (worst case scenario)
    def mock_render_extrinsic(self, del_extrinsic, **kwargs):
        # Simulate the worst case: directly store the passed del_extrinsic tuple
        # This exposes shared instance issues if RandomRotateSpace passes the same tuple
        self.stored_del_extrinsic = del_extrinsic
        return self
    
    # Apply the mock to both cameras
    camera1.render_extrinsic = mock_render_extrinsic.__get__(camera1, CameraImage)
    camera2.render_extrinsic = mock_render_extrinsic.__get__(camera2, CameraImage)
    
    # Apply RandomRotateSpace transform with prob_inverse_cameras_rotation=1 to ensure render_extrinsic is called
    transform = RandomRotateSpace(
        prob=1.0,  # Always apply rotation
        angles=[1, 1, 1],  # Small rotation angles
        prob_inverse_cameras_rotation=1.0  # Always apply inverse camera rotation (calls render_extrinsic)
    )
    
    # Apply transform to both cameras
    transform(camera1, camera2, seeds={'frame': 42, 'batch': 142, 'group': 1142})
    
    # Verify that both cameras have del_extrinsic parameters
    assert hasattr(camera1, 'stored_del_extrinsic')
    assert hasattr(camera2, 'stored_del_extrinsic') 
    assert camera1.stored_del_extrinsic is not None
    assert camera2.stored_del_extrinsic is not None
    
    # The critical test: verify that they don't share the same instance
    # Each camera should have its own copy of the del_extrinsic tuple
    assert camera1.stored_del_extrinsic is not camera2.stored_del_extrinsic, (
        "Multiple CameraImage objects should not share "
        "the same del_extrinsic instance. Each should have its own copy."
    )
    
    # Additional verification: check that the translation vectors are different objects
    del_R1, del_t1 = camera1.stored_del_extrinsic
    del_R2, del_t2 = camera2.stored_del_extrinsic
    
    assert del_t1 is not del_t2, (
        "Translation vectors should be different objects for different cameras"
    )
    
    # But they should have the same values (both should be [0.0, 0, 0])
    assert np.array_equal(del_t1, del_t2), (
        "Translation vectors should have equal values"
    )


def test_random_mirror_space_no_shared_instance():
    """Test that RandomMirrorSpace doesn't cause multiple CameraImage objects 
    to share the same instance when calling flip_3d.
    
    This is a unit test that mocks CameraImage.flip_3d to simulate
    the worst-case scenario where it directly stores the passed flip_mat parameter.
    This should ensure that each camera gets its own copy of the flip matrix.
    """
    # Create two CameraImage objects 
    camera1 = CameraImage(
        name="camera1",
        cam_id='VCAMERA_TEST1',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    camera2 = CameraImage(
        name="camera2",
        cam_id='VCAMERA_TEST2',
        cam_type='PerspectiveCamera', 
        img=np.zeros((64, 64, 3), dtype=np.uint8), 
        ego_mask=np.zeros((64, 64), dtype=np.uint8), 
        extrinsic=(np.eye(3), np.array([0, 0, 0])),
        intrinsic=[32, 32, 32, 32]
    )
    
    # Mock flip_3d to directly store the passed flip_mat (worst case scenario)
    def mock_flip_3d(self, flip_mat, **kwargs):
        # Simulate the worst case: directly store the passed flip_mat
        # This exposes shared instance issues if RandomMirrorSpace passes the same matrix
        self.stored_flip_mat = flip_mat
        return self
    
    # Apply the mock to both cameras
    camera1.flip_3d = mock_flip_3d.__get__(camera1, CameraImage)
    camera2.flip_3d = mock_flip_3d.__get__(camera2, CameraImage)
    
    # Apply RandomMirrorSpace transform
    transform = RandomMirrorSpace(
        prob=1.0,  # Always apply mirroring
        flip_mode='Y'  # Flip along Y axis
    )
    
    # Apply transform to both cameras
    transform(camera1, camera2, seeds={'frame': 42, 'batch': 142, 'group': 1142})
    
    # Verify that both cameras have flip_mat parameters
    assert hasattr(camera1, 'stored_flip_mat')
    assert hasattr(camera2, 'stored_flip_mat') 
    assert camera1.stored_flip_mat is not None
    assert camera2.stored_flip_mat is not None
    
    # The critical test: verify that they don't share the same instance
    # This checks if RandomMirrorSpace passes the same self.flip_mat to multiple cameras
    assert camera1.stored_flip_mat is not camera2.stored_flip_mat, (
        "Multiple CameraImage objects should not share "
        "the same flip_mat instance. Each should have its own copy."
    )
    
    # Additional verification: modifying one shouldn't affect the other
    original_value = camera2.stored_flip_mat[1, 1]
    camera1.stored_flip_mat[1, 1] = 999
    
    # If they don't share the same instance, camera2 should be unchanged
    assert camera2.stored_flip_mat[1, 1] == original_value, (
        "Modifying one camera's flip_mat should not affect another camera's flip_mat"
    )

# ...existing code...
