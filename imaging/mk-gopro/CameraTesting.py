import cv2
import ifaddr
import goprocam
for adapter in ifaddr.get_adapters():
    if "GoPro" in adapter.nice_name:
        print("Found gopro")
        for ip in adapter.ips:
            if ip.is_IPv4:
                addr = ip.ip.split(".")
                addr[len(addr) - 1] = "51"
                addr = ".".join(addr)

gopro = goprocam.GoProCamera.GoPro(ip_address=addr, camera=goprocam.constants.Camera.Interface.GPControl)

vid = cv2.VideoCapture(2)

while(True):
    ret, frame = vid.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()