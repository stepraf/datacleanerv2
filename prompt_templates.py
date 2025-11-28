"""
Prompt templates for AI information extraction.

This module contains prompt templates and formatting instructions
used for Azure OpenAI API calls in the AI Advanced tab.
"""

# Default prompt template for information extraction
# DEFAULT_PROMPT_TEMPLATE = (

#     "Extract product information from the following rows: [rows]. "
#     "Return a simple product name for each row."
# )

# JSON format instructions appended to prompts
JSON_FORMAT_INSTRUCTIONS = (
    "\n\nReturn ONLY a valid JSON array with this structure:\n"
    "[\n"
    '  {"row_id": 1, "product_name": "Product Name 1"},\n'
    '  {"row_id": 2, "product_name": "Product Name 2"},\n'
    "  ...\n"
    "]\n\n"
    "Important: Return valid JSON only, no additional text or explanation."
)

DEFAULT_PROMPT_TEMPLATE = """

You will receive a batch of product-description strings from a freight or logistics database.
Each description may or may not contain enough information to deduce a specific CPE product name.

Your task:

For each input description, analyze the text and determine whether it likely refers to a CPE device from known vendors and families 
examples include but are not limited to:

Vendor	Product Families / Model Series	Aliases / Naming Conventions	Product Examples
Arcadyan	VRV/VGV series gateways, GPON ONT series	VRV-xxxx, VGV-xxxx, Arcadyan ONT	Arcadyan VGV7519
Arris / Comcast (Xfinity OEM)	Arris TG/TS/CM DOCSIS gateways; Comcast Xfinity xFi gateways (XB6, XB7, XB8)	XB6/XB7/XB8, TGxxxx/TMxxxx, ISP OEM SKUs	Arris TG1682G, Arris TG2482A, Xfinity XB6, Xfinity XB7, Xfinity XB8
Askey	RAC2V series, 5G/LTE CPE	RAC2V, Askey 5G CPE	Askey RAC2V1K
AVM	FRITZ!Box (75xx, 66xx, 56xx, 55xx series), FRITZ!Repeater line, FRITZ!WLAN, FRITZ!Fon	FRITZ!Box <model>, FRITZ!Repeater <model>, FRITZ!WLAN, FRITZ!Fon	FRITZ!Box 7590 AX, FRITZ!Box 7530, FRITZ!Box 6690 Cable, FRITZ!Box 5590 Fiber, FRITZ!Repeater 3000 AX, FRITZ!Repeater 1200 AX, FRITZ!Fon C6
Cambium Networks	PMP 450 subscriber modules, ePMP subscriber CPE (Force 180/200/300), cnRanger LTE CPE, cnPilot routers, cnWave 60GHz CPE, ePMP Force CPE series, XV Wi-Fi, cnMatrix edge, 450/450b subscriber modules	PMP450, Force 180/200/300, cnPilot R, cnRanger CPE, Force 300/400, 450b, cnWave CPE, XV Wi-Fi	Cambium PMP 450b, Cambium PMP 450i, Cambium ePMP Force 180, Cambium ePMP Force 200, Cambium ePMP Force 300-16, Cambium ePMP Force 400C, Cambium cnPilot R190W, Cambium cnPilot R200, Cambium cnWave V5000, Cambium cnWave V3000, 1Cambium XV2-2 Wi-Fi AP, 1Cambium cnMatrix EX1028
Cradlepoint	IBR series (IBR600/IBR900), E300/E3000 enterprise gateways, R1900/R2100 5G, W-series 5G outdoor units	IBR600/IBR900/IBR1700, E300/E3000, R1900/R2100, W1850/W2005	Cradlepoint IBR600C, Cradlepoint IBR900, Cradlepoint IBR1700, Cradlepoint E300, Cradlepoint E3000, Cradlepoint R1900, Cradlepoint R2100, Cradlepoint W1850, Cradlepoint W2005
Ericsson	W-series fixed wireless terminals	W-series fixed wireless terminal	Ericsson W30 Terminal, Ericsson W35 Fixed Wireless Terminal
Huawei	EchoLife HG/HS/EG series gateways; HN series ONT; CPE Pro/Pro 2 (H112/H122); 5G CPE Win/Win2; B-series LTE CPE (B3xx/B5xx); EchoLife ONT (HG8xx/HG9xx/HS8xx/HS86xx); OptiXstar ONT (HN8xx/HN82xx/HN85xx); Wi-Fi 6 ONT (OptiXstar HN8546/HN8245Q/HN8145V)	EchoLife HG/HS/EG; HN ONT; CPE Pro; H112/H122; 5G CPE Win/Win2; B3xx/B5xx LTE CPE; B535/B818; HG8xx/HG9xx; HS8xx/HS86xx; HN82xx/HN85xx; OptiXstar <model>; Q/V family Wi-Fi 6 ONT	Huawei EchoLife HG8245H, Huawei EchoLife HG8546M, Huawei OptiXstar HN8245Q, Huawei OptiXstar HN8145V, Huawei CPE Pro H112-370, Huawei CPE Pro 2 H122-373, Huawei B525, Huawei B535, Huawei B818-263, Huawei 5G CPE Win
Inseego	FX2000, FX3100, Wavemaker FG series, MiFi mobile hotspot series	“FXxxxx”, “Wavemaker FG”, “MiFi <model>”	Inseego Wavemaker FX2000, Inseego Wavemaker FX3100, Inseego Wavemaker FG2000, Inseego MiFi 8000, Inseego MiFi 8800L, Inseego MiFi X PRO 5G
Kaon Media	KSTB series set‑top boxes	KSTB-xxxx	
Milesight	UR industrial routers (UR32, UR75), UG LoRaWAN gateways, cellular CPE	UR series, UG series, Milesight 5G CPE	Milesight UR32, Milesight UR75 5G, Milesight UG65 Edge Gateway, Milesight UG56
NetComm Wireless	NF/NL series fixed wireless CPE, 4G/5G ODU/IDU, NTC industrial routers	NFxx, NLxx, NTC-xxxx, IDU/ODU <model>	NetComm NF18ACV, NetComm NF18MESH, NetComm NL1901ACV, NetComm NTC-140W, NetComm NTC-40WV
Netgear	Nighthawk routers (R/AX series), Orbi mesh (RBK/RBR/RBS/RBE series), LAX/LM/LBR 4G/5G CPE, DOCSIS CM/CB/CAX series, Nighthawk M (mobile hotspot) series, LBR cellular gateway, Insight BR gateways	“Nighthawk R/AX <model>”, “Orbi RBK/RBE <model>”, “LAX20/LBR20”, “CM/CB-xxxx”, “Nighthawk <model>”, “AX/BE”, “Nighthawk M5/M6”, “Orbi <model>”, “RBK/RBR/RBS”, “CAX/CMxxxx”, “LBR-20”, “Insight gateway”	Netgear R7000, Netgear R8000, Netgear RAXE500, Netgear Orbi RBK753, Netgear LAX20, Netgear LBR20, Netgear Nighthawk M5 (MR5200), Netgear Nighthawk M6 (MR6150), Netgear CM2000, Netgear CAX80, 1Netgear BR500
Nokia	ONT/ONU series (G-010/G-240), Beacon Wi-Fi mesh series, FastMile 4G/5G CPE/gateways, 7368 ISAM ONT family	G-010/G-240, XS-01x, FastMile 4G/5G, Beacon <model>, 7368 ONT	Nokia G-010G-P, Nokia G-240G-E, Nokia G-240W-F, Nokia XS-010X-Q, Nokia 7368 ISAM ONT G-240W-F, Nokia Beacon 1, Nokia Beacon 6, 1Nokia FastMile 5G Gateway 3, 1Nokia FastMile 4G Gateway
Peplink	Balance routers; MAX BR/MAX HD cellular routers; MAX Transit series; Surf SOHO series	Balance <model>; MAX BR/HD; Transit Duo; Surf SOHO	Peplink Balance 20X, Peplink Balance 310, Peplink MAX BR1 Mini, Peplink MAX HD2, Peplink MAX Transit Duo, Peplink Surf SOHO MK3, Peplink BR1 Pro 5G
Sagemcom	F@ST broadband gateways; PON ONT/ONU; 4G/5G CPE; Video Soundbox/STB series	F@ST <model>; Sagemcom ONT/ONU; Video Soundbox	Sagemcom F@ST 5688W, Sagemcom F@ST 3890, Sagemcom F@ST 5560, Sagemcom Video Soundbox
Technicolor / Vantiva	Home Gateways (CGA/CGM/CGI series), DOCSIS Gateways (TC and TG series), xPON ONTs (FG series), Android/Operator STB (UIW/USW and KM7 series)	TC<model>, CGA/CGM/CGI-xxxx, TG-xxxx, FG ONT series, UIW/USW STB, KM7 Android STB	Technicolor TC4400 DOCSIS 3.1, Vantiva TG789vac v2, Technicolor UIW4001 STB
ZTE	ZXHN home gateways (F6xx/H2xx series), MC/MU 4G/5G indoor CPE, MF 4G routers, F41xx/F62xx ONT series, ZXHN F6xx/F8xx ONTs, F660/F680/F689 series, ZXHN H series gateways, ZTE B-series IPTV STB	"ZXHN <series>", "MC/MU 5G CPE", "MF <model>", "F41xx/F62xx ONT", "ZTE B-series STB", "MC801A/MC8020/MC888", "MU500", "MF289", "F6xx/F8xx", "F660/F680/F689", "ZXHN Hxxx series"	ZTE MC801A 5G CPE, ZTE MC8020 5G CPE, ZTE MC888 5G CPE, ZTE MU500 5G hotspot, ZTE MF289 LTE CPE, ZTE ZXHN F660 ONT, ZTE ZXHN F680 ONT, ZTE ZXHN F689 ONT, ZTE ZXHN H298A gateway, ZTE ZXHN H267A gateway, 1ZTE ZXHN F620 GPON ONT, 1ZTE B760H IPTV STB, 1ZTE MF286 4G router


If the description contains enough information to identify a specific model with high probability, extract:
1. Product name in standardized form: "<Vendor> <Model>"
   Example: "Huawei EchoLife HG8546M", "Netgear Orbi RBK753", "Cradlepoint IBR600C", "ZTE ZXHN F660"
2. Manufacturer/Vendor name: The company that makes the product
   Example: "Huawei", "Netgear", "Cradlepoint", "ZTE"

If the product cannot be confidently inferred, output empty strings for both fields.

Do not output explanations, reasoning, or uncertainty scores—only the extracted product name and manufacturer, or blank values if not found.

Extract product information from the following rows: 

[rows]."""