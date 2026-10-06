"""Exercise the actual worker method locally with a deterministic test session, not a neural model."""
import ast,io,math,time,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
from PIL import Image

source=ast.parse(Path(__file__).with_name('app.py').read_text())
worker=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='TopazEngine')
method=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='upscale')
method.decorator_list=[]
namespace={'math':math,'time':time}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'app.py','exec'),namespace)
upscale=namespace['upscale']

class IdentityScaleSession:
    def run(self,outputs,feed):
        tile=feed['image'];return [np.repeat(np.repeat(tile,4,axis=1),4,axis=2)]

class WorkerChecks(unittest.TestCase):
    def engine(self):
        info={'scale':4,'tile_size':128,'overlap':32,'input_noise':'noise','input_sharp':'sharp','input_comp':'comp','input_net':'image','output_net':'out','name':'Deterministic local test session'}
        return SimpleNamespace(sessions={'cgi_4x':{'sr':IdentityScaleSession(),'info':info}})
    def payload(self):
        img=np.zeros((121,173,3),dtype=np.uint8);img[:,:,0]=np.arange(173,dtype=np.uint8);img[:,:,1]=117;img[:,:,2]=43
        buf=io.BytesIO();Image.fromarray(img).save(buf,format='PNG');return img,buf.getvalue()
    def test_overlap_and_boundary_assembly(self):
        src,payload=self.payload();result=upscale(self.engine(),payload,bit_depth=8)
        out=cv2.cvtColor(cv2.imdecode(np.frombuffer(result,dtype=np.uint8),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
        expected=np.repeat(np.repeat(src,4,axis=0),4,axis=1)
        self.assertEqual(out.shape,expected.shape);self.assertLessEqual(int(np.max(np.abs(out.astype(int)-expected.astype(int)))),1)
    def test_16_bit_png(self):
        _,payload=self.payload();result=upscale(self.engine(),payload,bit_depth=16);decoded=cv2.imdecode(np.frombuffer(result,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.dtype,np.uint16);self.assertEqual(decoded.shape,(484,692,3))
    def test_missing_model_rejected(self):
        _,payload=self.payload()
        with self.assertRaises(ValueError): upscale(self.engine(),payload,model_name='standard',scale=2)

if __name__=='__main__': unittest.main()
