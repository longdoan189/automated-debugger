def thisfunction(x: str, y:str, z:str) -> str:
    val = ""
    try:
        num_y = ord(y)-ord("a")
    except:
        return ""
    for e in x:
        num_e = ord(e)-ord("a")
        val += chr( (num_e+num_y)%26 + ord("a")) #fix here
    return val 

inputfunc = thisfunction