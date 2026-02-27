#!/usr/bin/env python3
"""Profile PoC through OpenAI API servers.

This script starts multiple vLLM OpenAI API server processes and profiles
PoC /api/v1/pow/generate calls in parallel.
"""

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import torch

import vllm
from vllm.poc.env import (
    POC_PROFILE_DIST_THRESHOLD,
    POC_PROFILE_FRAUD_THRESHOLD,
    POC_PROFILE_P_MISMATCH,
)
from vllm.poc.protocol.runtime_types import Artifact
from vllm.poc.utils.validation import validate_artifacts

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(line_buffering=True, write_through=True)
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(line_buffering=True, write_through=True)

PUBLIC_KEY = "test_pub_key"
BLOCK_HASH = "TEST_BLOCK"
VALIDATION_SAMPLE = {
    "public_key": PUBLIC_KEY,
    "block_hash": BLOCK_HASH,
    "block_height": 2732723,
    "node_id": 1,
    "artifacts": [
        {"nonce": 0, "vector_b64": "6bIBLJ84d7S7KgK2kDOOs201Va3xMwq1"},
        {"nonce": 1, "vector_b64": "8rGdrQi3Vq58LTC09TEwMH42dDM5Khm5"},
        {"nonce": 2, "vector_b64": "VKw5Nh40yTcgNE0gdyxptzYlrLe4r4qu"},
        {"nonce": 3, "vector_b64": "NTVxKwYmsjdDMzMwAyybMdStQjfiNlM2"},
        {"nonce": 4, "vector_b64": "ByggMOa0WDQzMtOt0bXoOfwogjSrrawr"},
        {"nonce": 5, "vector_b64": "xJgTOZy3u7AQLFCsB7RRtG4zrh6bMMY1"},
        {"nonce": 6, "vector_b64": "S7jVr7A0o7SItjA3D6xuKhKnvLQVnQ20"},
        {"nonce": 7, "vector_b64": "5racsxKxXbZHtZKpoitTssSwbDR7OOOt"},
        {"nonce": 8, "vector_b64": "/jRANrQxADWfMNg2OzTnMvs0WSRAti4z"},
        {"nonce": 9, "vector_b64": "6qyEMBy2gi9KNyk4nDI1sZqybDY/pwc0"},
        {"nonce": 10, "vector_b64": "6zZJtciqzSkHNAW5Z7DJKgW0B7VzMzMs"},
        {"nonce": 11, "vector_b64": "uTIQNU058LHtJGUyiCQquNYwr60IMjgs"},
        {"nonce": 12, "vector_b64": "xrY7rWG0PbKBNH8kxLa9pKgxNDRqODgy"},
        {"nonce": 13, "vector_b64": "2jdKtCWw37BWNjkuLjbTrzc3Py8CNSWr"},
        {"nonce": 14, "vector_b64": "4LRvKLG0fC0CtsI2KbCdsaipFznLMR2t"},
        {"nonce": 15, "vector_b64": "7DCxL4U2vrDkMzM5BjDgNl00hyR6rJ2u"},
        {"nonce": 16, "vector_b64": "p7AKrwKzbLTGLNk4dbGsNseviSkvtNG2"},
        {"nonce": 17, "vector_b64": "y7UHMssuRaZ0MK61ljGHObwtJbIJMr+0"},
        {"nonce": 18, "vector_b64": "sDJ1NnqvjbPBtfE4H7ThqeQwU7AGtVcs"},
        {"nonce": 19, "vector_b64": "/a18Lz64QyGFtdO0MLPBMXwpN7TFNOW3"},
        {"nonce": 20, "vector_b64": "XTaBsNc2KjijJ7KslyywrVSZ/jK5tqU1"},
        {"nonce": 21, "vector_b64": "47J/Niwwl606s0O1QLUYsTQ29jJpNCE3"},
        {"nonce": 22, "vector_b64": "TzXhtD4jXThQtKezsza0sPomBauSNpSY"},
        {"nonce": 23, "vector_b64": "TDHDrAS0eioHohc5HbPxNK2pYrBNOJay"},
        {"nonce": 24, "vector_b64": "B7Jfp74v+q8nuPy2nzWcs0WxJjF3t9Qw"},
        {"nonce": 25, "vector_b64": "3DDoNPMmXLfbMIw29zFvrF8w2DQKuGY0"},
        {"nonce": 26, "vector_b64": "RTSzMzw5uzGXs38yNhpCOAgtrq2GqNkv"},
        {"nonce": 27, "vector_b64": "qLYwLoy0vrUWLGgscbWNraY4bbXhMMub"},
        {"nonce": 28, "vector_b64": "m7BHt442VzbNNQi1/qAhMco2AKh7LssX"},
        {"nonce": 29, "vector_b64": "LzF2tj4UWjX8MVU0Uzh0tPirurVtNDmv"},
        {"nonce": 30, "vector_b64": "Fa/xN/E1Oq3bLqww5C7isHI4ujWAtFYv"},
        {"nonce": 31, "vector_b64": "aDW5NYM25CwOOdau2DBVNAKYoDIFqusx"},
        {"nonce": 32, "vector_b64": "QzVbt/AtuTbwMSeuZjC0NQOuIrSft04r"},
        {"nonce": 33, "vector_b64": "vTGcMPUsnDJ8pvA0jTcJNDq5CTStpR0x"},
        {"nonce": 34, "vector_b64": "dTe2NwWtWq2HNwgwFi58tq0yEzQjoF6w"},
        {"nonce": 35, "vector_b64": "BaymrYywPbZ0tVum5bXKuJWvvKuAtEU1"},
        {"nonce": 36, "vector_b64": "MLjeM3MuU7hAsfY1shi9MP41IKNzMQOt"},
        {"nonce": 37, "vector_b64": "zLAyNLg2K7S7NOetwrA4taY2CDJvtw2w"},
        {"nonce": 38, "vector_b64": "DbhxsI2oMCwTsqQ5vDA5rkswUSgXNemx"},
        {"nonce": 39, "vector_b64": "iDXpMQetwqFrILopdTfTqoEyuTcWJrA4"},
        {"nonce": 40, "vector_b64": "2DjzKdg1pC1tLaU0cC/Dpi+yTjbEtCG1"},
        {"nonce": 41, "vector_b64": "JDadNkQxB7b/s6y1mLBXLu00y7SesX40"},
        {"nonce": 42, "vector_b64": "La2VNxO0drR1NJkpVze9r1SwJDFdss43"},
        {"nonce": 43, "vector_b64": "qDPWJB04FDjmsZotHrMmNbU0irSFMuyv"},
        {"nonce": 44, "vector_b64": "NjLmsQo0KLbbs5s1zqwluBS0FTZiLNKy"},
        {"nonce": 45, "vector_b64": "4zJ7N2CnEqrBL8Q1dbQyrrgy8bFKuOC1"},
        {"nonce": 46, "vector_b64": "lTHeLC+tfa02tToyq7dgtLC4VzYtnU6o"},
        {"nonce": 47, "vector_b64": "0C38NTizfjmPsSMuaxkQtkMzoi4Lsg6z"},
        {"nonce": 48, "vector_b64": "UTbeswGm3idXNEq4Yi22M6k3pzTmMdEt"},
        {"nonce": 49, "vector_b64": "VTGjNnQqRqo7rfGsWjidKYS1rLUWsJO3"},
        {"nonce": 50, "vector_b64": "g7jfMbs2NjTvto2ydiv9qhk1DLHVHWKy"},
        {"nonce": 51, "vector_b64": "GDBbtn0r7aomsRS3H7PnNPm16DQHNoE0"},
        {"nonce": 52, "vector_b64": "PZzUMR81kDczNICpZbKBM/KyqLWTMhY4"},
        {"nonce": 53, "vector_b64": "6bB8Mzs34Srqr/WxkLbXNH0u3Tf5tJcz"},
        {"nonce": 54, "vector_b64": "rzNspZG2qa9ntP0uTbe+r9Q2MrdgMRSy"},
        {"nonce": 55, "vector_b64": "0zTKsa0q0DIlsGE5iagVtV+tfbXTNTij"},
        {"nonce": 56, "vector_b64": "gjUJMVK4zrOdNM+k+7PPsAYsgLQJOKAb"},
        {"nonce": 57, "vector_b64": "FjEQGjW5O7ACqNgxVyMmNZ23XTIxtOkz"},
        {"nonce": 58, "vector_b64": "5K6hsCe3BjjWrCOqRjZEMVwskDPAKhk4"},
        {"nonce": 59, "vector_b64": "zrQhrz835rGZsjOssbOutzqyJzanNUyv"},
        {"nonce": 60, "vector_b64": "zjUEt2Gzg7eFNTCwpK1MsZApLDWDtNMy"},
        {"nonce": 61, "vector_b64": "6LRxtskrjbZNMCm1ZSQ1ptStmbYTtEQ3"},
        {"nonce": 62, "vector_b64": "t7SGNGqzpTXFNQemjTMIN5W0OzKSsmQ1"},
        {"nonce": 63, "vector_b64": "16+cMZk0yTeFsfQpPLfNsvw1bLZTs6cb"},
        {"nonce": 64, "vector_b64": "OK1JODepgbUiOF4w9bAGM32xfrUKsZ6z"},
        {"nonce": 65, "vector_b64": "eLOBtqu3VbI3swU2ljITt6ovCyhjsiqx"},
        {"nonce": 66, "vector_b64": "Np31sIexnyg2N+g42yoeLs0vHjhvstUx"},
        {"nonce": 67, "vector_b64": "CbdyNxmxA6pOswk2ibQcsyW1t7VkrPIq"},
        {"nonce": 68, "vector_b64": "LbPqrHI0ay0guaawdTEcsi43jC9vNbWx"},
        {"nonce": 69, "vector_b64": "tC6BtKYZmzPDNUA4DatfNd8kAjTIsne3"},
        {"nonce": 70, "vector_b64": "jTjWrX4xFLcPtaGoU6xkMCm0dCaUNby1"},
        {"nonce": 71, "vector_b64": "CDUtubOuG7aysq8xwrWFtAmxOanQL3Ge"},
        {"nonce": 72, "vector_b64": "1bDbLFi0bCsnuL62K7UlNzaoFzTKM0uw"},
        {"nonce": 73, "vector_b64": "ebdjtDa4DjZ0tTQzxLFIse+pWixhs5Sg"},
        {"nonce": 74, "vector_b64": "wTUQLn8ymC5FMFYyhjfymdw3C7BELxc4"},
        {"nonce": 75, "vector_b64": "LrOhMvY0Fif5If4ynrght0W0k7V4K+6x"},
        {"nonce": 76, "vector_b64": "hC+tsA83VKExMBSwRi6FKHWyxDJQJGe6"},
        {"nonce": 77, "vector_b64": "pDWKqqiljjj0tN81GCD3pFKqeTi3qrcd"},
        {"nonce": 78, "vector_b64": "VbMLL2K46ahQsQOxGyWftHo0iDjOIQk1"},
        {"nonce": 79, "vector_b64": "8zJgsiWr2bhhq8wqbLieJWGyQrSDtQEp"},
        {"nonce": 80, "vector_b64": "RjSbsWqs/KTJruIxRjJuNv+lATaCOcox"},
        {"nonce": 81, "vector_b64": "eab/sTi0TjUrKsI5fzAKsPmu1TCKNjsx"},
        {"nonce": 82, "vector_b64": "YzSVtPemkLSpOI011y3SMJKgRy1zpRm4"},
        {"nonce": 83, "vector_b64": "1rV7N2SysTABslo4fDMVsNg266TaoEOp"},
        {"nonce": 84, "vector_b64": "nje5MLc0qijVLog4tzB5pTi1uapcsaO2"},
        {"nonce": 85, "vector_b64": "oC26rHSwu7ZeNV2woa9tNVk4/jUntEwx"},
        {"nonce": 86, "vector_b64": "JTRAtBEzyjVkKXKxCrm8MNQyrrEBM0M1"},
        {"nonce": 87, "vector_b64": "T7cWoKM1STO5MaAbIa9dspMuH7gLOKmi"},
        {"nonce": 88, "vector_b64": "9Sp8MHY44TYqMfm20bG3LZEyuaxfNRK0"},
        {"nonce": 89, "vector_b64": "GjWUsaUv165/Mek37qs2smq3Pji6rCGx"},
        {"nonce": 90, "vector_b64": "yLaaLeMyjqwbt34ezjAjrpE4xbXRq0+0"},
        {"nonce": 91, "vector_b64": "vLCKI2kwhrZHNV62ZCa5NCahVbP7t+g1"},
        {"nonce": 92, "vector_b64": "YjTTs7e2/awsNJowsrhdMgaugzQ2sfM0"},
        {"nonce": 93, "vector_b64": "pa+GrXAcR7L/LayxIDTmrPAwZDlDNdw3"},
        {"nonce": 94, "vector_b64": "IzjntcO0LqzKqa6sUDcrsr6uRCzONm+z"},
        {"nonce": 95, "vector_b64": "SbbHNDU06Cs6NMK3YTjWKSAtMbRnJuEo"},
        {"nonce": 96, "vector_b64": "BzHEIOQ45zZPMYo1uy5qNMUyv7NyM+uw"},
        {"nonce": 97, "vector_b64": "PLGQOIWvJDVMNS2yATV7LWmw5DTTJ4O2"},
        {"nonce": 98, "vector_b64": "QaxDMSK4Ka1VriO0Ei7KtVGwZ66Xt5k3"},
        {"nonce": 99, "vector_b64": "cKRdOTk0Yyw2MK80IbfuLpuyiBvVs92z"},
        {"nonce": 100, "vector_b64": "UTCcL3oziS0aOJqpYqyWrES0aKvuOWyt"},
        {"nonce": 101, "vector_b64": "H7iBMWqz2zDDneWsJSj3s8KdT7Bws5U5"},
        {"nonce": 102, "vector_b64": "vquOm04uaSrJtgSwITW6s+42ardyNgE0"},
        {"nonce": 103, "vector_b64": "7jKAtBO3xigfK3cmprFat+gumjehrPG2"},
        {"nonce": 104, "vector_b64": "4TXFtXAlBbiipkOttSyJrGKxJ7VSuHOy"},
        {"nonce": 105, "vector_b64": "06XiriewkTA5tQ01+rfwNhi0BrKXNao0"},
        {"nonce": 106, "vector_b64": "dy7zLSqxo7UXN9U20KiLNku13bVjMe+u"},
        {"nonce": 107, "vector_b64": "LrgKLaSrDTg6slA0HbEfNTy2niwYMBqz"},
        {"nonce": 108, "vector_b64": "+TS+N6+23rb1s80nlS+knUUzSSr6stI1"},
        {"nonce": 109, "vector_b64": "mjO9MDs4t62vrrk1fjQcNfUxTKk2OHAn"},
        {"nonce": 110, "vector_b64": "cCRMNzK0R7HSNHCuY682sq+tgbdftTg3"},
        {"nonce": 111, "vector_b64": "DbQ6uOUzjxygMJagJzhftlEwtrMlNC2r"},
        {"nonce": 112, "vector_b64": "Z7YFnWQt1bGirnmthzGVNvOvGjR3Nfa4"},
        {"nonce": 113, "vector_b64": "LTAHNia3GLZctSE0a7ZlnTuxJy1fsz80"},
        {"nonce": 114, "vector_b64": "zjC1LTCz8CluuPoz/LPmLKO4caUftHE0"},
        {"nonce": 115, "vector_b64": "bTSNtIwzVa2dLau1VK0FthQwQbYMuFi0"},
        {"nonce": 116, "vector_b64": "ja/FsLms6TLnL680r7LsNEisJLfiNkO4"},
        {"nonce": 117, "vector_b64": "aTewNZOs5zLdtEc1wLe9rTqwubVSsOMT"},
        {"nonce": 118, "vector_b64": "gqr5K2e2i7P3Gr0tO7YCrXy0qzjitBC1"},
        {"nonce": 119, "vector_b64": "rbSKMne0IzLaNBKzejCOM8S44DMaMFG1"},
        {"nonce": 120, "vector_b64": "bK8qtIAwESzUs6Q4LzY5NIg3MTGyMWIf"},
        {"nonce": 121, "vector_b64": "KTEjM9Y1yqyfN3wuYDNzOKiypKhNKP+1"},
        {"nonce": 122, "vector_b64": "LzV3rQ24PjjcMuowEK1CNti0jKlPMV2h"},
        {"nonce": 123, "vector_b64": "y7h7riUsZ7GbLjyy9DCpNsqv/KxPsWm4"},
        {"nonce": 124, "vector_b64": "sbeGskWyEDdSLZe3CLVtsgcqPLRoso+u"},
        {"nonce": 125, "vector_b64": "9TOKrx6u3zV7o2ko8KYFuB8ujLnvKtmx"},
        {"nonce": 126, "vector_b64": "wC1ZLAw0kjOOLQq33rTtKHC5T7RLLwMx"},
        {"nonce": 127, "vector_b64": "iDG5LKY3JDWLL5CuWzdiNTgxgjRctRs0"},
        {"nonce": 128, "vector_b64": "JbTtM+64C7XJLGOwE6rTNsIoFi6jKp22"},
        {"nonce": 129, "vector_b64": "i6yJM1QfHTXhKZM4RysntuMtRTXrMUO3"},
        {"nonce": 130, "vector_b64": "IiqgKKG4PbflNAkzgrNQrBm1NLWtMR2w"},
        {"nonce": 131, "vector_b64": "TzUiKqkssTXHNGS0tjRqtEWtYxzwrxo5"},
        {"nonce": 132, "vector_b64": "RCqYr8YqnDH6r7c2eie0trm3GDjpMyqx"},
        {"nonce": 133, "vector_b64": "GDUbNk4pta8QOnMwAq40HjctqjTBsYuu"},
        {"nonce": 134, "vector_b64": "sKwxOcKxwrSSrPqz6bIyNrqtPDCyNHC0"},
        {"nonce": 135, "vector_b64": "K6wmrKA1+DYDtLs3wLbsJCE0WLUOsSuk"},
        {"nonce": 136, "vector_b64": "aacEtxy0C6raL2e1GzdLqV0pZzV7tss1"},
        {"nonce": 137, "vector_b64": "sDB8OGMpYS7isYKxyjFlsdKiuzmnr/ce"},
        {"nonce": 138, "vector_b64": "SjXBNO+qsjRlqDeu8Tayttw3+zAeNIut"},
        {"nonce": 139, "vector_b64": "FjNpNd2ozzXbtq2uKqzftQUriTUCqfA3"},
        {"nonce": 140, "vector_b64": "ETSLquez37fEFMCzvCx1N1O0My4pNxu0"},
        {"nonce": 141, "vector_b64": "UrX2tsy1Brg2r8e0RSxmMzWrebGSNVwq"},
        {"nonce": 142, "vector_b64": "4rWWsQ609rQQMSe4DC0YOCIoyBfzNKwv"},
        {"nonce": 143, "vector_b64": "IbRbMTO1Xqh4tuOpSre6Hxo0f7DutTQ3"},
        {"nonce": 144, "vector_b64": "xTGaNEI0ZTUlMo61fTWUr4m3/rIWs9+0"},
        {"nonce": 145, "vector_b64": "PqlgLDI0zrIDML201q/1uQc2dLFzqeMx"},
        {"nonce": 146, "vector_b64": "KrckKkywaawbJI+zDK2BMBUzJrFpOHu4"},
        {"nonce": 147, "vector_b64": "+DGXrtyzI7VkM5oSXbb4sDKxqLUkNWS4"},
        {"nonce": 148, "vector_b64": "/q9stPS2lzAFtSW2yLi2Mu2mhabXpxyz"},
        {"nonce": 149, "vector_b64": "a6WLJJwuezLTMMmlBzYwM6+4CTXMtky1"},
        {"nonce": 150, "vector_b64": "KLM8rGQvdTktMSMsNrVSsziwMSzUsls3"},
        {"nonce": 151, "vector_b64": "Vq5prZs46SwMthiogjZVsjyr9bcEqc0y"},
        {"nonce": 152, "vector_b64": "GzWwJzGuxLX1tgq3frLfrcS3LixrtF8u"},
        {"nonce": 153, "vector_b64": "6qCRMXyi/SwLNROq+TXBNC+y7blyKecy"},
        {"nonce": 154, "vector_b64": "Zix9s6uwkTTCtD+lOrPmMwA0cjJuML45"},
        {"nonce": 155, "vector_b64": "4LX0sHy2tzV2tgyiMrIYNdgzGyw4tW40"},
        {"nonce": 156, "vector_b64": "/bKFqZe0xTTKsy8xwDKtuJW0pCS4Nf+0"},
        {"nonce": 157, "vector_b64": "pDTPM+005DCiNG21CzhNrTw39SzPs2Kt"},
        {"nonce": 158, "vector_b64": "MKUvFOm0861ArYUxDDp8LP00x7RztIiw"},
        {"nonce": 159, "vector_b64": "xLH3M/etATSZuL2xiqO4M+wgBTLLOF8w"},
        {"nonce": 160, "vector_b64": "GTSGNM84qC1iMO4mfLAoN+a1ezK8JDU0"},
        {"nonce": 161, "vector_b64": "ITNXNc4lirbzNfw3NDQ1sFcsvzQwMMC0"},
        {"nonce": 162, "vector_b64": "hKQiN2UqhS6LOB65uKraqRsw9jFAoxWb"},
        {"nonce": 163, "vector_b64": "ybKJtFQjrbdkL1ioxzYTOdqxW6WULzql"},
        {"nonce": 164, "vector_b64": "BLHttbUw1zXtM9MvcK9iMfE3qzYZNhYj"},
        {"nonce": 165, "vector_b64": "sTdOt4yzEjP7LjAvPrUerTc3orTHrBCx"},
        {"nonce": 166, "vector_b64": "6DYONAK0E7SHtHW0RrE6KGMx0jU+LCq4"},
        {"nonce": 167, "vector_b64": "sTM+sD01UrdDsoa33TNBMNsw4jNoM9i1"},
        {"nonce": 168, "vector_b64": "GbaJNQy0PzWhNg40Rrher/gw3SvJpcOm"},
        {"nonce": 169, "vector_b64": "Ji4jtMg3p7hwswgPTbPrsxUx6C/Brde1"},
        {"nonce": 170, "vector_b64": "OLDxt1K4jjSzNdSvm6rPLDId7Z+GNNc1"},
        {"nonce": 171, "vector_b64": "bDb/Mnsx07jkNcyt8bGatZEugC5tNCak"},
        {"nonce": 172, "vector_b64": "D632sBa2zrgPNBcwvDBBtlqzxzEvq481"},
        {"nonce": 173, "vector_b64": "tTJAt8A1mzUqpNKzGiZGMAu0hTfdtW8o"},
        {"nonce": 174, "vector_b64": "E7QvOUU2ZRteLR2fjjGlLucyBTaSsqG0"},
        {"nonce": 175, "vector_b64": "7aTVLNsykS+gOFA41TfqLJUtu6+8Lzyx"},
        {"nonce": 176, "vector_b64": "DDifqdm2cTTNs14zf7J4NW81w6znM4Wx"},
        {"nonce": 177, "vector_b64": "ECV4NMKzi7HPuMqvvTGSLBi2KbantYUo"},
        {"nonce": 178, "vector_b64": "gyhtNzm2vSwsrT6yfDbUN4yt7SyGNday"},
        {"nonce": 179, "vector_b64": "3qivOAA1OK3WtX0mALI3NVY2brQ+K2My"},
        {"nonce": 180, "vector_b64": "XrXtMJI4yK8xq0y0NJ5aLU604jISo4q4"},
        {"nonce": 181, "vector_b64": "da2eruKtZDmzshGyFjDjMGI2UzISNzyt"},
        {"nonce": 182, "vector_b64": "ObYiMrUyKCWJq4i0q7UinFs3YTNbMDO4"},
        {"nonce": 183, "vector_b64": "JrWFNxqw66tqKki49bJ5lTo3UjUurGWl"},
        {"nonce": 184, "vector_b64": "OKcZtDwgMC7RI8CzbrFJtYO1FLPNOe6x"},
        {"nonce": 185, "vector_b64": "ZbAlMrK4NbKIt9ys/K0DOCOvuzE2Lziw"},
        {"nonce": 186, "vector_b64": "DLW8KvEkGbGfOdy0+iDrNK2wVyzYMok1"},
        {"nonce": 187, "vector_b64": "cy+cOAwtFDK8MaimVzeCtiO20i/uLAQy"},
        {"nonce": 188, "vector_b64": "2baTsLY1CLU8Mh60hy0JuD+gQyaqMNU2"},
        {"nonce": 189, "vector_b64": "U644rRI4YjDwNwu0vLM0Nbc0eTVZsZel"},
        {"nonce": 190, "vector_b64": "hDFWsNo2ILReNx+4aK8vri2yHzRjrda0"},
        {"nonce": 191, "vector_b64": "rbScNqgy8zZvtIQ23jRqsLU0dqAopz+0"},
        {"nonce": 192, "vector_b64": "FjC9E3i3Fi2gtJOsabVtNdgw8ziusLCv"},
        {"nonce": 193, "vector_b64": "7LdcNSA5sDTErYsnVbXpLFYsFS9AKX6p"},
        {"nonce": 194, "vector_b64": "+bXqNAsuT7bHMem017U3Miu4xp2drFCx"},
        {"nonce": 195, "vector_b64": "kTGnNJuyeB9BuKG1pzSfs1Sve6BDMBy4"},
        {"nonce": 196, "vector_b64": "ZriyMfSnJrW2sh00W7V5NQYxrLb1peCw"},
        {"nonce": 197, "vector_b64": "17GspzW2GDEEuCIBFahrsEut+DN0OPk1"},
        {"nonce": 198, "vector_b64": "MSG0LYarQ7OgHGS5Tjb5MH4zUrC7rYw3"},
        {"nonce": 199, "vector_b64": "8a12rmerjzKXuYE1IrcBmTMtsDQiFT4y"},
        {"nonce": 200, "vector_b64": "rDBVKRO3FjZ/smO14B6rM282VTNVLxw3"},
        {"nonce": 201, "vector_b64": "ey5YM5o1EKcFOEYyzS0qt3w0o7EWt+Et"},
        {"nonce": 202, "vector_b64": "4TQjNxk2/jATtS8ygrW9IjqwlzO0MzK2"},
        {"nonce": 203, "vector_b64": "2DJkNGg0HLQdtkKwj6Q7N7wyU7e2tGWx"},
        {"nonce": 204, "vector_b64": "UzNbs1k1w60KsPY0qzYFtL42QbKKnle3"},
        {"nonce": 205, "vector_b64": "ligquIu28DVNtbY1LjQqMhKuQrM2qVIw"},
        {"nonce": 206, "vector_b64": "r7Kas5Yywq4muBe2WSvVsne2ObH8rPC2"},
        {"nonce": 207, "vector_b64": "rTFzrVy5Pi6PrCGt5qbRsmk4YK+yNUOp"},
        {"nonce": 208, "vector_b64": "NrJnMU8wBizNt102cioJNDU4qTABLCm2"},
        {"nonce": 209, "vector_b64": "STjXtf21UDVCMVswLK9GsbcyfS8Nrcs2"},
        {"nonce": 210, "vector_b64": "xyn4s7w21bQUMsu2/DTRKv6uobL1KVe4"},
        {"nonce": 211, "vector_b64": "TrRJtIK0wCwKNyo4hDLuLi6wsykTN0uy"},
        {"nonce": 212, "vector_b64": "Y7GZtrwyFba1tPgvSLYNJgW0w7dXMl2w"},
        {"nonce": 213, "vector_b64": "FLTpLIm3NLQ4Lqot7B2nrFq2JC+2L0G5"},
        {"nonce": 214, "vector_b64": "668dLJiyqqctOFg3YTMjtay0kDA2toey"},
        {"nonce": 215, "vector_b64": "xbQJMZ6odKUKMuQ3X7TEqB4pbbW6uAy0"},
        {"nonce": 216, "vector_b64": "mbAbrBA4+LUMNhWx46GONXExGjCMNm80"},
        {"nonce": 217, "vector_b64": "tCVUop2v97YxuC8huLZ3qk439rDCLgS1"},
        {"nonce": 218, "vector_b64": "7TNpL0YwF7D7seWyUbp/KvAzRSaQtaAn"},
        {"nonce": 219, "vector_b64": "mLHvt7CwmjMYOL+ytqOONPkkrqoJuGik"},
        {"nonce": 220, "vector_b64": "4y1SqGKuWa69OCkwNjFsL4A4SLA0NyMx"},
        {"nonce": 221, "vector_b64": "eS7uN2WxULU6sBi34DESNmCw2KcFtfW0"},
        {"nonce": 222, "vector_b64": "z7IsOMUyyi5Tr4G4UbSPI2g1BbA6M56y"},
        {"nonce": 223, "vector_b64": "nTGLNvW0Q7P9rVCkQjZ6t8w0VLZWrlUw"},
        {"nonce": 224, "vector_b64": "0q+FtRO2KqvIqQkzkClBMmup1DhCNUk2"},
        {"nonce": 225, "vector_b64": "36gRKPKxhi6+JlArP7BiuA+3/bT+t8o0"},
        {"nonce": 226, "vector_b64": "Jzhls7OwXDBEN0u0NDb/LiEywDQptDIs"},
        {"nonce": 227, "vector_b64": "W7KHNHq1EK4Hrj8zVjixs362crQTtBoy"},
        {"nonce": 228, "vector_b64": "kyu4rHY2ejdHtoC3HDbWMqQvjCJIKHQx"},
        {"nonce": 229, "vector_b64": "oq5QNBq4+TFGLh83xLXNpOYsvDMarGE3"},
        {"nonce": 230, "vector_b64": "XTLMrJK4krQFtdm15a6aMCi00LGsNPo0"},
        {"nonce": 231, "vector_b64": "4TGAsCe1NzQepym3KbEwMVg0N6oSOZOy"},
        {"nonce": 232, "vector_b64": "CDVRtMA0ibOdsnep9riQsJiuV7LuNQWz"},
        {"nonce": 233, "vector_b64": "WTRzs3Kzf6o3LRoxmbdWNkM1oTRPN2el"},
        {"nonce": 234, "vector_b64": "O7OFsqk4gjIpK/2xSKJQNOy1RDdQtCec"},
        {"nonce": 235, "vector_b64": "z7c+rKUzGzObMy0sMTdQqEg1YTSSKye3"},
        {"nonce": 236, "vector_b64": "O60nsXA0jbE5OSy2SzOYtSmm17Oopi+z"},
        {"nonce": 237, "vector_b64": "WbHSsaIxUajdprE48DPTtoIxCbZLqxM2"},
        {"nonce": 238, "vector_b64": "crXtr2qpprUMOGI0eLUVMCU0P7dEqVms"},
        {"nonce": 239, "vector_b64": "MTYluKaxU7P5ID81KTELuN+wqS7usm+x"},
        {"nonce": 240, "vector_b64": "6yuzMzuyGrUuNYG2Srh5pv8zIzXxMXWy"},
        {"nonce": 241, "vector_b64": "I7i7I/+3wS4iKdc1gC2WpEO1tzXyqNo0"},
        {"nonce": 242, "vector_b64": "9LPdsDI2Qa+RNsmyGa2ssOotvC0JJKA5"},
        {"nonce": 243, "vector_b64": "ISgLtNG2KjFTM704L6xSMHyty7ZgtWeo"},
        {"nonce": 244, "vector_b64": "5DTyOH0zhrQMMM+06bSarUioYjLutNuy"},
        {"nonce": 245, "vector_b64": "MzEqMYs3Qa+0tIKxirVhORquz6sOrk2q"},
        {"nonce": 246, "vector_b64": "pCgaNX216rOhtHo3SCeCtfW3ZzBPqW4x"},
        {"nonce": 247, "vector_b64": "1LlDKColyC+aMeyxGaR1uBSqp7CfMuQu"},
        {"nonce": 248, "vector_b64": "2jVduRKuOLQfNlc0RKW0rROzMTFIJ+mx"},
        {"nonce": 249, "vector_b64": "rLc9Nk02FC++MZywlbUgNYe0k6xPtAuw"},
        {"nonce": 250, "vector_b64": "6LCtp2I2mLQnNZY2YSxzNbu1arOdtQAy"},
        {"nonce": 251, "vector_b64": "xTJ9LSQqEauQtEK0Ji6fNVk2ESdWMX+5"},
        {"nonce": 252, "vector_b64": "KzTRMBi3NjCdtzc4wLFIs5KuIbK8LVWz"},
        {"nonce": 253, "vector_b64": "ra8OM8UnLTkNNBEtKbKbL8ErPjYXMJI3"},
        {"nonce": 254, "vector_b64": "JjNCtQosNzDRNJ8wPyRQuF8yTKm/OBqx"},
        {"nonce": 255, "vector_b64": "HLXGNZm2vTR2rxC2PDELtKQz0TakrkQv"},
        {"nonce": 256, "vector_b64": "mikutIiuhi/PsUc5jTVKr/O3ZDMpKHaq"},
        {"nonce": 257, "vector_b64": "OzhQNi2mRzibsT2xqq8UtM6vaKwDte0w"},
        {"nonce": 258, "vector_b64": "MSwvMPG3VbApKr+1qDcurtYz0Cz8pzc4"},
        {"nonce": 259, "vector_b64": "DK1HMDYqCDM6tea3Hqwit/KwmKz0NT03"},
        {"nonce": 260, "vector_b64": "l7QtNVY18bGusGAxYDXUsaU3rTdDrtWr"},
        {"nonce": 261, "vector_b64": "tDd/sH2xvrDjOHe2zrMaLfanDil5sFA0"},
        {"nonce": 262, "vector_b64": "iDAetEIxObEMMQ+45bAeLti0tjYlOEUu"},
        {"nonce": 263, "vector_b64": "pzB6OJoxWrc6rD6uIrRuswY0FzIgs/81"},
        {"nonce": 264, "vector_b64": "HbcPsyi1i7b6MRKoDKp+MVOxE7DeuGso"},
        {"nonce": 265, "vector_b64": "czDuLDmwOaqgqVAywLF3teMv2zpIrRgu"},
        {"nonce": 266, "vector_b64": "G6g5NNIxNy57sY4vUqwctyq4NjhDs48y"},
        {"nonce": 267, "vector_b64": "QzAGrlS0nC1ONTG4Ebmzs48yrKKRICyt"},
        {"nonce": 268, "vector_b64": "wbRtNJ4y27FQsXm2mbCbtkY0XqqGM/W3"},
        {"nonce": 269, "vector_b64": "gbVltuG0jrX1JbCw/jZXtRq1VbT0J8wt"},
        {"nonce": 270, "vector_b64": "/KyqNMQygjULtH60KbRGN40xHTXqM+M1"},
        {"nonce": 271, "vector_b64": "gzB0tB823rIetNQeI7ndpYS1VLSksFIw"},
        {"nonce": 272, "vector_b64": "ejDEuY2dEzMrNtwqBjQ1LdU1UawGs6Es"},
        {"nonce": 273, "vector_b64": "V63RsQkuXjVhOFgkLLekLzK1HzLsNLi0"},
        {"nonce": 274, "vector_b64": "MDbFLEUnwq8jtGo00Ko3NiY2EKsrt882"},
        {"nonce": 275, "vector_b64": "yTRFsgc04LVyLUIvTa+JOMO16JrupvU2"},
        {"nonce": 276, "vector_b64": "j7Q6Mj4wbTMXOcUy1gNusViuf7I8NFY3"},
        {"nonce": 277, "vector_b64": "Mzg4KxYtXLg6My21ZrGBHWU3vK4UKxAt"},
        {"nonce": 278, "vector_b64": "1TCHACo3l7FHNRIxwyy1MMyxUzQxObWz"},
        {"nonce": 279, "vector_b64": "E7ZTtIGx+rImpA2tD7B+Ncsk+7TjiXm5"},
        {"nonce": 280, "vector_b64": "MrCvsLExfzFupmywLjGHOFEzWjnesWGi"},
        {"nonce": 281, "vector_b64": "MLS5mjE01i73NZu3LDY1tRutTrTMMkE1"},
        {"nonce": 282, "vector_b64": "1CmOuSue9DE7LeqwajYDqmqxfDecsWYu"},
        {"nonce": 283, "vector_b64": "gS/rNVAkpzjnMNC4nS4sMFkwkDNcJtiw"},
        {"nonce": 284, "vector_b64": "z7QqNdezTjR4sb04prQrrRWztipoNGA0"},
        {"nonce": 285, "vector_b64": "4rXpr+mpmKlKKlc5VClFOIsqQLEpNIov"},
        {"nonce": 286, "vector_b64": "QDLisEy4jTQtNGc2XzfFrT8sXjPWskew"},
        {"nonce": 287, "vector_b64": "mLYhMwak1zYQtYIwPbStMIm19Cs/M1q3"},
        {"nonce": 288, "vector_b64": "ZDB1tLSwdqwfNjgyPbLgsrA2fKy1sBw5"},
        {"nonce": 289, "vector_b64": "NivIOJwiAjS4tiQ1CDOwLfO1iaXFr760"},
        {"nonce": 290, "vector_b64": "BbXnMyY1tTVCtRq30TCUK7s3T69wrRKx"},
        {"nonce": 291, "vector_b64": "WasAtZqswDHaM+K3LbnFrskfLrT2MBwy"},
        {"nonce": 292, "vector_b64": "0TPArxKu+bdGNkix+zF/Muy2W7OeNroi"},
        {"nonce": 293, "vector_b64": "WzUlK/CkGzhptXU1PLVHq3S3m7M5rSSl"},
        {"nonce": 294, "vector_b64": "XTE/sGmuT7bDOEC11TN5Moq2GzOTLDUs"},
        {"nonce": 295, "vector_b64": "irdiMZ+wlDL8Nr6scTLmtIwwCy1hOJcx"},
        {"nonce": 296, "vector_b64": "ErUwOHIxmLQttKK0lLQwq7g0dC3UNnmn"},
        {"nonce": 297, "vector_b64": "VLTtsNQzqareN5amhzcKNFQ11DHZsB62"},
        {"nonce": 298, "vector_b64": "s7QqMuCynzdFnGg2b7ATuBYv3DTeM2Kg"},
        {"nonce": 299, "vector_b64": "0qgTNI83wLQzN7qpr7TjpAc2rbFMNnwr"},
        {"nonce": 300, "vector_b64": "dLAZtZ00hjWqtVu0q6YgOP8xZLCXtf4x"},
        {"nonce": 301, "vector_b64": "YTNgKPSxbbnYsDqvkTa/MR2wtS6DNEA1"},
        {"nonce": 302, "vector_b64": "Qahcsh8yb7SntFqoY6mStQs5EK69M6W2"},
        {"nonce": 303, "vector_b64": "TbOrrga47LX+sCi09rUntUKz/LM/tGsx"},
        {"nonce": 304, "vector_b64": "6rXDOQaxwymWsqW07qwfJF2oNK9ONgSw"},
        {"nonce": 305, "vector_b64": "8LQHtfI25bUqMjOz27CKMHqp/zYqso81"},
        {"nonce": 306, "vector_b64": "hTAXOK40DrcVKZArObQftgquobWaNHWq"},
        {"nonce": 307, "vector_b64": "vbE2uOqtL7Jtsq4xvCznNQg4FbIetVUy"},
        {"nonce": 308, "vector_b64": "6LIcNiC1dzFrMXcrHbMlMBg5GzY2rMEw"},
        {"nonce": 309, "vector_b64": "KLolNKwwCKvRqF0y9alcMxq1BKwCtE0z"},
        {"nonce": 310, "vector_b64": "YbUmtV81cTK4MHe4NrSQMlo2WDHpJBoZ"},
        {"nonce": 311, "vector_b64": "lTQgOPGuC5tUNpSpFbfZsCw3lLHCLfiw"},
        {"nonce": 312, "vector_b64": "D7GlqJwrEaZ2tCG3tLQBuSk2xi9jsjyv"},
        {"nonce": 313, "vector_b64": "XatMLLUzdTbVtP4tSTaOtpk4gSW6JVAw"},
        {"nonce": 314, "vector_b64": "5DP+tXqrup/2LWMifCw0Okk1dK/mMOwx"},
        {"nonce": 315, "vector_b64": "iSXDtVqjui/pNLewSKtSuJ823rZDsjUz"},
        {"nonce": 316, "vector_b64": "9zTSrO0z7rHRrBG0yjRJMH8zI7RwOBE3"},
        {"nonce": 317, "vector_b64": "wyXEN2GvBapfr8W0t7QcuSc1FKv7MJcw"},
        {"nonce": 318, "vector_b64": "hLZYtrw3USv6LVImTDRZtPyz7TQitQmx"},
        {"nonce": 319, "vector_b64": "WbN7OJg01iuWMQuwl7PcLfMwrjSTrIk4"},
        {"nonce": 320, "vector_b64": "eDKpsTk2Q6hQtAC3wjOVLR6t3rfstlsu"},
        {"nonce": 321, "vector_b64": "JjZJsOApl7LUNL0xCLakpzC5TLAkLXy0"},
        {"nonce": 322, "vector_b64": "r7NkMYwexTakMkw2gLScsvExwDeFrua1"},
        {"nonce": 323, "vector_b64": "l7W1qCEmTyy9NsYwfzadOB4uKjBqqxc2"},
        {"nonce": 324, "vector_b64": "ETQuMhUwf7bitcC3ITQtNPW0XiuMNL4x"},
        {"nonce": 325, "vector_b64": "RrAyNL+iVzeKLrao8bS8udyrlqp1MvMx"},
        {"nonce": 326, "vector_b64": "uzC4pdI4uTSwM5kfx7gFrCo0Jqmprbqx"},
        {"nonce": 327, "vector_b64": "IyyZpEm1SToSsICqm7OwKoC2kafxsPuo"},
        {"nonce": 328, "vector_b64": "GS2IuBE04i4QqqMwA6h+OGC2MKblNBms"},
        {"nonce": 329, "vector_b64": "JCUQMvW2yjDlHJ00trnLLAUwFTF+NOwx"},
        {"nonce": 330, "vector_b64": "gLIeq5w0y6WoM8w43DVhKR6y8qzyJzQ4"},
        {"nonce": 331, "vector_b64": "ljQ8Ksq1u6rHIVI1BC9uNa61WK7oOLMx"},
        {"nonce": 332, "vector_b64": "oTDiLky4g6qmsJywhbcqMHs3dzXmLI+z"},
        {"nonce": 333, "vector_b64": "qzRPq+yuJbjBLz+w7yzoI0Iwz7bQuLox"},
        {"nonce": 334, "vector_b64": "8DLHtduuZbAjNtwzKDS5tZex+amxuDYv"},
        {"nonce": 335, "vector_b64": "Bi1jtaSfibgfuOKscqz/tXAo065jNSyw"},
        {"nonce": 336, "vector_b64": "xjd4spEwiSt3thmdGbXtM7Uw6TD+tK83"},
        {"nonce": 337, "vector_b64": "YrBSNu6yJCRwtICnfzjstDYz/LU0sU80"},
        {"nonce": 338, "vector_b64": "ISE3oBUuGC/tJzC3kDAKNMyx9bgCMtU3"},
        {"nonce": 339, "vector_b64": "WbSfpnKwHRC5trQk0LdAOdYzFKFckC+m"},
        {"nonce": 340, "vector_b64": "XbSpNrUgEzjksV80lzDqtZU1L62hspQz"},
        {"nonce": 341, "vector_b64": "EypyssI2lDjKNDkzACwQM9yw4bBRt9kq"},
        {"nonce": 342, "vector_b64": "G7gUMqkwTzhdNDCuGSiNqlo2WCsStXky"},
        {"nonce": 343, "vector_b64": "nLBrs+IpEDewuPKx/rRXrRe1rDWSp1ww"},
        {"nonce": 344, "vector_b64": "0Cgpt8Ug+7TpsgImWjPtONU2siyDMWis"},
        {"nonce": 345, "vector_b64": "yyX5LVe0dajeNzCyfKh0Lmy3rjhqslqx"},
        {"nonce": 346, "vector_b64": "SaSpMjk3U7FmNOGxHB69tOo4bTBktJkz"},
        {"nonce": 347, "vector_b64": "BbljtLa1fjSls4mwkbVOscIvhLNdrxUw"},
        {"nonce": 348, "vector_b64": "ETbMNZI0Y667tAsxLjAMNOC1qrTANNE1"},
        {"nonce": 349, "vector_b64": "L7bEsE202DkPJ0M1y6YkLdwytym8nsAz"},
        {"nonce": 350, "vector_b64": "UDnBMp+wA68KJF6u97TstDEp3TYisrIz"},
        {"nonce": 351, "vector_b64": "ozc8t2Msx7UpKcgwuDAsLWiwQDRRNcw2"},
        {"nonce": 352, "vector_b64": "DDSmMiu1GrQGs8m5ki0QpzAtJbXWsIku"},
        {"nonce": 353, "vector_b64": "rzVgpEWxPjOOrPe2wLPsMi+2tS2AMIO4"},
        {"nonce": 354, "vector_b64": "KrhTMwEwJDSpKsox07LJtBouRDeqtKC1"},
        {"nonce": 355, "vector_b64": "aimiN78tKzIdNVg1ZTAut5I0vzG8NiYk"},
        {"nonce": 356, "vector_b64": "BrLNJhQ1LbRfCCowqjG4NFAjZDmfNuqx"},
        {"nonce": 357, "vector_b64": "pzTlMnGw/jT2tWUwQ7FdMtos8LQ1ucMr"},
        {"nonce": 358, "vector_b64": "LjTUtNo3iDGeK1E4tTFdMvQvVDA+M4S1"},
        {"nonce": 359, "vector_b64": "m7b0sq8u17YqOKI1YLJBMGk1/q6Mpxeu"},
        {"nonce": 360, "vector_b64": "CDWJLGih4q7ir9Ky87U8uG+1NrYIMJ41"},
        {"nonce": 361, "vector_b64": "u7cDNha1BbRONfQ0Oy3mMHaxPDanMpUo"},
        {"nonce": 362, "vector_b64": "TrdduNmqmrAroQ8riycONlO0vjHbsqS2"},
        {"nonce": 363, "vector_b64": "SjadtZ8v5raZsPi2LDNzM580nqhmtRau"},
        {"nonce": 364, "vector_b64": "zTP+NM+1/TdLKjYq7LEENbC2nLOpNOot"},
        {"nonce": 365, "vector_b64": "jq2oslyivbjBtl+0kzbuL2w1YC5IKfWx"},
        {"nonce": 366, "vector_b64": "ua0znhQ4GjTPuO4wWrfRLt6sLK0eI+Qy"},
        {"nonce": 367, "vector_b64": "WzN6Nwi4165wtBCyyrKuNbmqzrPqNNKw"},
        {"nonce": 368, "vector_b64": "kTcpNC800rAhNQO2kjQrNcw2tqyPqU6i"},
        {"nonce": 369, "vector_b64": "GbRkIfu2wCqBsGc0SDc2KOww0SgTOXiu"},
        {"nonce": 370, "vector_b64": "JTFyNR+w3zAyqkU20TMysCm1LLizsoa2"},
        {"nonce": 371, "vector_b64": "4TVPNYg3hzYwrSa117QXNkiotKulLoMs"},
        {"nonce": 372, "vector_b64": "NTi7tKShprCjsaq2oKylM1k3pjREMzao"},
        {"nonce": 373, "vector_b64": "rq4VMjEsgSzKOLA1eLMttqcwrLSFsMA1"},
        {"nonce": 374, "vector_b64": "HDQyMU03+bK/OQEz+6I5qcixky3OsrEl"},
        {"nonce": 375, "vector_b64": "RzGcNomwzLGdtXMkYirsM/MtPTlvtAWz"},
        {"nonce": 376, "vector_b64": "BTZhLzmr2aGqMkm2WDhWNIi1tK0VLQm2"},
        {"nonce": 377, "vector_b64": "hyy8NkEucy6QGE6prLEbsTQvqjUUNNw5"},
        {"nonce": 378, "vector_b64": "PTUwM6Eg9SFSL0owozE4MWQ4oDfitmoy"},
        {"nonce": 379, "vector_b64": "JK+lsHawqzHRMOO4SjSItBUzJSZCL2y4"},
        {"nonce": 380, "vector_b64": "HLcbMRSxFzf+Nds0hjWEr4Uvca6Ks1M1"},
        {"nonce": 381, "vector_b64": "tLfesuiy2THfMTu4Jq5gMo6wPDfbr2sy"},
        {"nonce": 382, "vector_b64": "ra6HN0c1azIbqcY3b7GPNvC1ibDHMBas"},
        {"nonce": 383, "vector_b64": "n7PTtJ4y/DDvNqS3HLZOqyY3yaooMASo"},
        {"nonce": 384, "vector_b64": "BzFMtFCqzDhBtOK3D7YmMfAqkavLsIcx"},
        {"nonce": 385, "vector_b64": "gLjkNb42KCoNlzUy1SyatcuukDPbJtw1"},
        {"nonce": 386, "vector_b64": "W6sQLy818LBYqOC42rZotZA0BiSKMKQ0"},
        {"nonce": 387, "vector_b64": "ha4fNkC1Z4LiNEKprTFOuXqqTrRPMooz"},
        {"nonce": 388, "vector_b64": "lLSPtHA1bbNStSs2MjSwNUM1hy7PNEus"},
        {"nonce": 389, "vector_b64": "RbVzN+W2lqj7MRo3l6/RMJSvZbW6NBQs"},
        {"nonce": 390, "vector_b64": "+TS9s5Syei6jMNYZwjHGtP+5XDGLsGqy"},
        {"nonce": 391, "vector_b64": "lyzBsFy1Q7CiN6GmRjRALECwPjh7N7ew"},
        {"nonce": 392, "vector_b64": "BzihNMAwurKbMsW0CrHILhO4krFItCu0"},
        {"nonce": 393, "vector_b64": "ujJUMduw2bGCLqczz7SEOAG3SazbMYS2"},
        {"nonce": 394, "vector_b64": "V7XeMzonHjT3sjq07zD9OIMznq2ftKw0"},
        {"nonce": 395, "vector_b64": "7Ir/LG25RLVnNmG0wrDksKq0wymyM8kh"},
        {"nonce": 396, "vector_b64": "7rOrNUCuHLRsOFm4DK7PJJW0J6jyMJkw"},
        {"nonce": 397, "vector_b64": "ErO7Jj60SLf2pwM4jiwJs+6y3jE4t500"},
        {"nonce": 398, "vector_b64": "fDRiOd2uPjKgts8mZaxuqBGubiTftPk1"},
        {"nonce": 399, "vector_b64": "/7KDqbczHDPDOf6xzi0BNoktyy7CtNEx"},
        {"nonce": 400, "vector_b64": "yDZRt+slVjRUm4Ixm7PDq7o2frScNEc1"},
        {"nonce": 401, "vector_b64": "WaZQpf4vly+QKSOxo7UVO3WwDarQKJ8u"},
        {"nonce": 402, "vector_b64": "NydDtcE4bSviqfMyjbZztnutNKDyLAg2"},
        {"nonce": 403, "vector_b64": "37COL6OwSjnUGaI2BKwZr3YJi7Swt6es"},
        {"nonce": 404, "vector_b64": "0DPzMsuxMrAgODK4UDBqtKg0sbHbMbqz"},
        {"nonce": 405, "vector_b64": "4pnSLVK4iq0sNUcxSTa8Mzu2g6iRNVo0"},
        {"nonce": 406, "vector_b64": "kDetNionEbAmNHesiqnyuAQ1hajvr7Sv"},
        {"nonce": 407, "vector_b64": "jzlftv0qoaqRtaMhLrFGrjYzpzQOrXKz"},
        {"nonce": 408, "vector_b64": "x6pmt2Y0PC4lru4uLLmDr5QvQrRGKl62"},
        {"nonce": 409, "vector_b64": "zivqq6gvxjRUoju45SnGtHu4GzFqtQ80"},
        {"nonce": 410, "vector_b64": "qLLrMmG327HXNRsiVLQLtHsz+ye6tyu1"},
        {"nonce": 411, "vector_b64": "mKYcLJ+z6jjVOEo0FaJ6r98sxy+Usdaz"},
        {"nonce": 412, "vector_b64": "zilRJho3sDLVL9SxWzbYK405dpUFmBk0"},
        {"nonce": 413, "vector_b64": "SbV+MOOxTSiYKXQ4Ya2wsAQ42J9vt2Kx"},
        {"nonce": 414, "vector_b64": "r6PWuEWavqa+KdqnMrAetHi4MjBusr+2"},
        {"nonce": 415, "vector_b64": "PDX7NMyt8acXtpgzt7MVthwwIDXytzWw"},
        {"nonce": 416, "vector_b64": "CjR3LBO6lLQ5MVsmWye6K2YuRbRhNh6p"},
        {"nonce": 417, "vector_b64": "L61wKb00YjLfsaqoViwbmIg4czROOOs1"},
        {"nonce": 418, "vector_b64": "JjBwMI+2DrJmtActhLIzslswu7kftM4o"},
        {"nonce": 419, "vector_b64": "xC6psgE1K6/ktR61jjRBsTS18zCguBqx"},
        {"nonce": 420, "vector_b64": "n622Neow0jYCL8OoMLUbqWi556/7Lckz"},
        {"nonce": 421, "vector_b64": "c7YWOLCeC6oauM4yCS/9MFywIrf2LrCl"},
        {"nonce": 422, "vector_b64": "UDCPrzMz760wunYx6LFAr9M0wrUpHJQv"},
        {"nonce": 423, "vector_b64": "mKuwtGMtETJpsiyvizKwNPq0Qbe2KvK4"},
        {"nonce": 424, "vector_b64": "ODVvOCy45zVKKIkwXzN9KDgpDxn2tJ2s"},
        {"nonce": 425, "vector_b64": "5jgRODMmmLPBtPaoaSlrqWilRjYcsNYy"},
        {"nonce": 426, "vector_b64": "c7jKsx6lEjMzLlgq4jJ8MEo4ITWlskY0"},
        {"nonce": 427, "vector_b64": "y7VhroyvwLLmts4rriqxLCS40S4iuKOz"},
        {"nonce": 428, "vector_b64": "SicDn4uxvTAlshK4JrU/HnK4T7UvtQIx"},
        {"nonce": 429, "vector_b64": "kLRbuA60A7UmNaoxIrL1M6G1bzMzrxyy"},
        {"nonce": 430, "vector_b64": "R7jlNGut4LIEsm6zeq2vuGUlwiS2NUAi"},
        {"nonce": 431, "vector_b64": "kajqMrwsOLKNM8EyzzNmNkynEjrzrSus"},
        {"nonce": 432, "vector_b64": "0jZKsuw2TrWFqQWypjGKrugt5bG7t7g1"},
        {"nonce": 433, "vector_b64": "gjM+NVG5Tq43tI4y37RrtfEtODBkrfix"},
        {"nonce": 434, "vector_b64": "YzTiOHewWjRDNSe08C/3Ld+yl7M4LEA2"},
        {"nonce": 435, "vector_b64": "fDTos4s0Uy3ytcY0ybUlMXU1TrC+ta41"},
        {"nonce": 436, "vector_b64": "vzYMJSatvLFCuH43QrSwr2osKCmIsWC2"},
        {"nonce": 437, "vector_b64": "dp22N6g4NzePLWStorT/si0iZiZGsOQz"},
        {"nonce": 438, "vector_b64": "B7d7NIewlzLrODau/qxVLgg07LR6rf+0"},
        {"nonce": 439, "vector_b64": "Ura0LDy3SjZxs+0z27S/MTi05LASthMr"},
        {"nonce": 440, "vector_b64": "KDMxtiq07zVzLVIy9TWIKQGs2zjvsd8o"},
        {"nonce": 441, "vector_b64": "yLFQOJ8zgC6PtfmvQS/itI41PLL2tF+1"},
        {"nonce": 442, "vector_b64": "SrC0r8OxCjL0NO04SraKKHy3ErKvrPOt"},
        {"nonce": 443, "vector_b64": "SS6Ht8o0+LGXNi4sLLUbsCg4NDNQsbwp"},
        {"nonce": 444, "vector_b64": "QzaCMK2tNLGFN1c2x62otoO0pzWZrbMw"},
        {"nonce": 445, "vector_b64": "oyyYsKC2+izNuf2xPSvfKLGz26zAII42"},
        {"nonce": 446, "vector_b64": "tSwbOLS1tzAjNXCyMbBbskw3+zERIPo1"},
        {"nonce": 447, "vector_b64": "Ni1nLqU4DDFdNakgR7F7sTc3rTexr2Cs"},
        {"nonce": 448, "vector_b64": "nK8VsFoygLEjtUs1Xbl6s8iy6jE3L9y0"},
        {"nonce": 449, "vector_b64": "Ii7KsD63gLRcNSE5D7Tfo+ohgDQqsHOm"},
        {"nonce": 450, "vector_b64": "Qq9fsAC0xTAUNug3OTSINFA1GamotUc1"},
        {"nonce": 451, "vector_b64": "yq26IaA1JLDwMi4yhDhUJT84ODYYrL8u"},
        {"nonce": 452, "vector_b64": "YzLCtHA0A67rsHS2AKG+qcSkajkzrdS1"},
        {"nonce": 453, "vector_b64": "ojbPM0Ww7LBkNHKq2y6TtKo3l7eqNCEw"},
        {"nonce": 454, "vector_b64": "YDW1FXmxpTF/t3e1UrPXM3W1erD2tB42"},
        {"nonce": 455, "vector_b64": "SbALqUI1CbarM4upZrJlOU2zvrHNsDe0"},
        {"nonce": 456, "vector_b64": "Z7ZfMyy0bC8hsYazCjX4uGW07bEDMJev"},
        {"nonce": 457, "vector_b64": "9TOEsCwlMrM/s02t+Lfvsty3UzT7tjUu"},
        {"nonce": 458, "vector_b64": "XbKstC82RTTKq0MyszEBuFy0KLMqpkI3"},
        {"nonce": 459, "vector_b64": "IK9SMP0yI7j2Nfc2IDhOK+iroKwgMs2u"},
        {"nonce": 460, "vector_b64": "HrVYMpgxXC9aNYI0pjSwONu1+C3xsgkt"},
        {"nonce": 461, "vector_b64": "07ZSLFgzXraHNoM1WbS9NNQ0DjEvsmgq"},
        {"nonce": 462, "vector_b64": "tqo/tLKplrKiMtyzebSttz84srWkq8Wz"},
        {"nonce": 463, "vector_b64": "ObiiMX+uMbkpMMot5S8gsGS0UTAstHSy"},
        {"nonce": 464, "vector_b64": "vzRUsjilgS6wJgA1oDQ6OcqrwTeOrGYu"},
        {"nonce": 465, "vector_b64": "nLYGtTA0Ni36N7k1ja7ktZey+bAgNC6h"},
        {"nonce": 466, "vector_b64": "ZTUFNEw0czRuNl4zADNgMEcxBKget4g2"},
        {"nonce": 467, "vector_b64": "Kjjqsa6oJTK6MUO2KjWRsz84wyS7MJ4t"},
        {"nonce": 468, "vector_b64": "g7hRLYwguTBiNam4drWbsdEqJKvYMlEp"},
        {"nonce": 469, "vector_b64": "KreTsGS4SzH0tmYvA6w0tsOiBTKmsz8o"},
        {"nonce": 470, "vector_b64": "668ONLMudbG+L/OrMjY8tTK4oxVlNXK3"},
        {"nonce": 471, "vector_b64": "rC4CtLm55bUSMTU12CahKbKsvbNxrVs0"},
        {"nonce": 472, "vector_b64": "nDLMOLanC7MJK4k2L7TVsN4wZLdTsgAv"},
        {"nonce": 473, "vector_b64": "kbPiNpQ2HiiDtAQ34rSWrVio1LD7tiAq"},
        {"nonce": 474, "vector_b64": "lS/+NQ048bXLL2WrObBVOHI0rhuRsewv"},
        {"nonce": 475, "vector_b64": "ADBBrTo6CDXXMHQ1P6hOsQAxwDKlL70w"},
        {"nonce": 476, "vector_b64": "FzbSsJMpfbN7uZe0BLGEKueqFbRDNLqz"},
        {"nonce": 477, "vector_b64": "N7GuMHqyjLF6OBqkvrPXsLOvyDQmOGE1"},
        {"nonce": 478, "vector_b64": "mbTRLCUqOzKdOBo4PDaRNG6tI6nJMXCn"},
        {"nonce": 479, "vector_b64": "ErczMhk0nbfjtE6xniwMOHik2DB1sDwz"},
        {"nonce": 480, "vector_b64": "d7Xxtea1ETaUrtKuPjadqwy4qDG7JBwm"},
        {"nonce": 481, "vector_b64": "YTH3MDK2d7mFtvgv0rMeMS0yNrDnqkat"},
        {"nonce": 482, "vector_b64": "ECfbt0Ez3jIxsiU2hjg4qeMssLXIsLgc"},
        {"nonce": 483, "vector_b64": "XDGgNLo4vLMLL5WvHrKwtwKt3SKwNiQt"},
        {"nonce": 484, "vector_b64": "KrjmMEsrVqvCqNS4aDXBtnmaZqYhsg4s"},
        {"nonce": 485, "vector_b64": "iCRcLCMsCyBfsZmw77PWs8o1bzQCqz+6"},
        {"nonce": 486, "vector_b64": "US9KOcS1RyzzsZooHqGxILUtMDiOLcu0"},
        {"nonce": 487, "vector_b64": "E7T1KFA2ALjONLitFTUcHdQ0t7bpMtmv"},
        {"nonce": 488, "vector_b64": "P6M6tRkwHLC2tLo4ArGeqTG35LSSLtq0"},
        {"nonce": 489, "vector_b64": "ZDLPrhshMjN7tUm4ZrkNKJCt0i2XsI+o"},
        {"nonce": 490, "vector_b64": "uK0vK7G2FTX+NIYw6rdtN20n4DRyMUyv"},
        {"nonce": 491, "vector_b64": "PLC+rcS1LLUarks1PrJsNrw45bBEJQsz"},
        {"nonce": 492, "vector_b64": "LLTqrP+xHzb4Li2uFrEHMY42HK7INde4"},
        {"nonce": 493, "vector_b64": "0zNNNeU2OixTuOqv+S2BLio0ODb0rMU0"},
        {"nonce": 494, "vector_b64": "TLZtMWc1TColqaUt/re+p2w0IjHXtk42"},
        {"nonce": 495, "vector_b64": "EbFYNs8tvrEbOeExWzUHtZSu4bI8Mpyx"},
        {"nonce": 496, "vector_b64": "v671KI2xxaxgtro35aiyNVyp87IZMrW4"},
        {"nonce": 497, "vector_b64": "xK44MBSvzLYmqQAx0bbELPG086oZOe8z"},
        {"nonce": 498, "vector_b64": "cbFqNzqy9TcxNwkpji84MX0vjLZAHeIy"},
        {"nonce": 499, "vector_b64": "d7ZItZctOTKeLqO1cTJ0N0K1oi05sSa2"},
    ],
    "encoding": {"dtype": "f16", "k_dim": 12, "endian": "le"},
}

SERVER_STARTUP_TIMEOUT_SEC = int(
    os.environ.get("POC_PROFILE_SERVER_STARTUP_TIMEOUT_SEC", "900")
)
SERVER_STARTUP_PROGRESS_SEC = int(
    os.environ.get("POC_PROFILE_SERVER_PROGRESS_SEC", "5")
)
BASE_PORT = 8766


def _resolve_project_root() -> Path:
    """Best-effort repo root resolution.

    When this script is executed from a shallow path (e.g. copied to
    `/e2e_poc_tiny.py` in a container), `Path(__file__).parents[3]` is invalid.
    We instead search upwards for a folder that looks like the vLLM repo root.
    """
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "vllm").is_dir():
            return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists() and (cwd / "vllm").is_dir():
        return cwd

    return script_path.parent


PROJECT_ROOT = _resolve_project_root()

_SERVER_LOG_TAILS: dict[int, deque[str]] = {}
_BASE_URL_TO_SERVER_IDX: dict[str, int] = {}


def _stream_server_logs(
    server_idx: int,
    stream: Any,
    log_file: Any,
) -> None:
    for line in iter(stream.readline, ""):
        text = line.rstrip("\n")
        if not text:
            continue
        print(f"[server-{server_idx + 1}] {text}")
        with contextlib.suppress(Exception):
            log_file.write(line)
        _SERVER_LOG_TAILS.setdefault(server_idx, deque(maxlen=120)).append(text)
    with contextlib.suppress(Exception):
        stream.close()


def _load_validation_payload() -> dict[str, Any]:
    return VALIDATION_SAMPLE


def _build_validation_map(payload: dict[str, Any]) -> dict[int, str]:
    artifacts = payload.get("artifacts") or []
    return {int(a["nonce"]): a["vector_b64"] for a in artifacts}


def _run_validation(
    computed_artifacts: list[dict[str, Any]],
    validation_map: dict[int, str],
    *,
    dist_threshold: float,
    p_mismatch: float,
    fraud_threshold: float,
    k_dim: int,
) -> dict[str, Any]:
    artifacts = [
        Artifact(nonce=int(a["nonce"]), vector_b64=str(a["vector_b64"]))
        for a in computed_artifacts
    ]
    stats = validate_artifacts(
        artifacts,
        validation_map,
        dist_threshold=dist_threshold,
        p_mismatch=p_mismatch,
        fraud_threshold=fraud_threshold,
        k_dim=k_dim,
    )
    return {
        "n_total": stats.n_total,
        "n_mismatch": stats.n_mismatch,
        "p_value": stats.p_value,
        "fraud_detected": stats.fraud_detected,
        "mismatch_nonces": stats.mismatch_nonces,
    }


def _build_device_slice(server_idx: int, tp_size: int) -> str:
    start = server_idx * tp_size
    end = start + tp_size
    return ",".join(str(i) for i in range(start, end))


def _build_device_slices(tp_size: int, api_server_count: int) -> list[str]:
    visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    required_devices = tp_size * api_server_count
    if visible_device_count < required_devices:
        raise RuntimeError(
            "Not enough visible GPUs in this environment for requested topology: "
            f"need {required_devices} GPUs for api_server_count={api_server_count}, "
            f"tensor_parallel_size={tp_size}, but only {visible_device_count} visible. "
            "If running in Docker, check --gpus and container visibility. "
            "Or lower api_server_count/tp_size, or set POC_PROFILE_DEVICE_SLICES "
            "explicitly."
        )

    return [_build_device_slice(i, tp_size) for i in range(api_server_count)]


def _normalize_device_slice(value: str) -> str:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty device slice")
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"invalid CUDA device id: {part!r}")
    return ",".join(parts)


def _parse_device_slices_env(value: str) -> list[str]:
    normalized = value.replace("|", ";")
    slices = [s.strip() for s in normalized.split(";") if s.strip()]
    if not slices:
        return []
    return [_normalize_device_slice(s) for s in slices]


def _resolve_topology(
    *,
    tp_size: int,
    api_server_count: int,
) -> tuple[int, int, list[str]]:
    """Resolve (tp_size, api_server_count, device_slices) for this environment.

    - If `POC_PROFILE_DEVICE_SLICES` is set, use it (semicolon-separated slices).
      Example: "0;1" for 2 single-GPU servers, or "0,1;2,3" for 2 servers w/ TP=2.
    - Otherwise, auto-downshift the requested topology to fit visible GPUs.
    """
    device_slices_env = os.environ.get("POC_PROFILE_DEVICE_SLICES", "").strip()
    if device_slices_env:
        slices = _parse_device_slices_env(device_slices_env)
        if not slices:
            raise RuntimeError("POC_PROFILE_DEVICE_SLICES is set but empty")

        inferred_tp_size = len(slices[0].split(","))
        if any(len(s.split(",")) != inferred_tp_size for s in slices):
            raise RuntimeError(
                "POC_PROFILE_DEVICE_SLICES must use the same number of devices "
                "per slice"
            )

        if tp_size != inferred_tp_size:
            print(
                "[INFO] overriding tp_size="
                f"{tp_size} -> {inferred_tp_size} from POC_PROFILE_DEVICE_SLICES"
            )
            tp_size = inferred_tp_size
        if api_server_count != len(slices):
            print(
                "[INFO] overriding api_server_count="
                f"{api_server_count} -> {len(slices)} from POC_PROFILE_DEVICE_SLICES"
            )
            api_server_count = len(slices)

        return tp_size, api_server_count, slices

    visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_device_count <= 0:
        raise RuntimeError(
            "No CUDA GPUs visible. Set CUDA_VISIBLE_DEVICES or run on a GPU machine."
        )

    if tp_size > visible_device_count:
        print(
            "[INFO] lowering tp_size="
            f"{tp_size} -> {visible_device_count} to fit visible GPUs"
        )
        tp_size = visible_device_count

    max_servers = max(1, visible_device_count // max(1, tp_size))
    if api_server_count > max_servers:
        print(
            "[INFO] lowering api_server_count="
            f"{api_server_count} -> {max_servers} to fit visible GPUs"
        )
        api_server_count = max_servers

    return tp_size, api_server_count, _build_device_slices(tp_size, api_server_count)


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _read_log_tail(log_path: Path, max_lines: int = 80) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return "<unable to read log file>"
    if not lines:
        return "<log is empty>"
    return "".join(lines[-max_lines:]).rstrip()


def _extract_root_cause(log_path: Path, window: int = 120) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return "<unable to read log file>"

    if not lines:
        return "<log is empty>"

    keywords = (
        "ValueError",
        "RuntimeError",
        "AssertionError",
        "CUDA out of memory",
        "out of memory",
        "invalid device",
        "WorkerProc failed",
    )
    for idx, line in enumerate(lines):
        if any(keyword in line for keyword in keywords):
            start = max(0, idx - 40)
            end = min(len(lines), idx + window)
            return "".join(lines[start:end]).rstrip()

    for idx, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            start = max(0, idx)
            end = min(len(lines), idx + window)
            return "".join(lines[start:end]).rstrip()

    return _read_log_tail(log_path, max_lines=120)


def _wait_for_health(
    server_idx: int,
    proc: subprocess.Popen,
    port: int,
    timeout_sec: int,
    log_path: Path,
) -> None:
    started = time.time()
    next_progress_ts = started + SERVER_STARTUP_PROGRESS_SEC
    while time.time() - started < timeout_sec:
        if proc.poll() is not None:
            tail = _read_log_tail(log_path)
            raise RuntimeError(
                f"API server {server_idx + 1} exited before becoming healthy "
                f"(port={port}, exit_code={proc.returncode}).\n"
                f"Last log lines ({log_path}):\n{tail}"
            )
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        now = time.time()
        if now >= next_progress_ts:
            elapsed = int(now - started)
            print(
                f"  waiting server {server_idx + 1} on :{port} "
                f"({elapsed}s/{timeout_sec}s)..."
            )
            next_progress_ts = now + SERVER_STARTUP_PROGRESS_SEC
        time.sleep(1)

    tail = _read_log_tail(log_path)
    raise TimeoutError(
        f"API server {server_idx + 1} on port {port} did not become healthy "
        f"in {timeout_sec}s.\nLast log lines ({log_path}):\n{tail}"
    )


def _start_server(
    server_idx: int,
    model: str,
    tp_size: int,
    device_slice: str,
    port: int,
    max_model_len: int,
    log_path: Path,
    log_file: Any,
    start_delay_sec: int = 0,
) -> tuple[int, subprocess.Popen, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = device_slice
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else f"{PROJECT_ROOT}:{existing_pythonpath}"
    )

    if start_delay_sec > 0:
        print(f"  delaying API server {server_idx + 1} start by {start_delay_sec}s...")
        time.sleep(start_delay_sec)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "--tensor-parallel-size",
        str(tp_size),
        "--max-num-seqs",
        "32",
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        "float16",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--enforce-eager",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=PROJECT_ROOT,
        start_new_session=True,
        text=True,
        bufsize=1,
    )

    if proc.stdout is not None:
        threading.Thread(
            target=_stream_server_logs,
            args=(server_idx, proc.stdout, log_file),
            daemon=True,
        ).start()

    print(
        f"  starting API server {server_idx + 1} on :{port} "
        f"(CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})"
    )

    try:
        _wait_for_health(
            server_idx=server_idx,
            proc=proc,
            port=port,
            timeout_sec=SERVER_STARTUP_TIMEOUT_SEC,
            log_path=log_path,
        )
    except Exception:
        _stop_process(proc)
        raise

    return server_idx, proc, str(log_path)


def _run_forward_api(
    base_url: str,
    model: str,
    block_hash: str,
    public_key: str,
    nonces: list[int],
    seq_len: int,
    k_dim: int,
    batch_size: int,
) -> tuple[float, list[dict[str, Any]]]:
    payload = {
        "block_hash": block_hash,
        "block_height": 2732723,
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": model,
            "seq_len": seq_len,
            "k_dim": k_dim,
        },
        "batch_size": batch_size,
        "wait": True,
    }

    attempts = 4
    last_error: str | None = None
    for _ in range(1, attempts + 1):
        t0 = time.time()
        response = requests.post(
            f"{base_url}/api/v1/pow/generate",
            json=payload,
            timeout=300,
        )

        if 200 <= response.status_code < 300:
            elapsed = time.time() - t0
            body = response.json()
            artifacts = body.get("artifacts")
            if artifacts is None:
                raise RuntimeError(f"No artifacts in response: {body}")
            return elapsed, artifacts

        response_text = response.text.strip()
        detail = response_text
        try:
            response_json = response.json()
            detail = str(response_json.get("detail", response_json))
        except ValueError:
            pass

        last_error = (
            f"HTTP {response.status_code} from {base_url}/api/v1/pow/generate: {detail}"
        )

        server_idx = _BASE_URL_TO_SERVER_IDX.get(base_url)
        if server_idx is not None:
            tail_lines = list(_SERVER_LOG_TAILS.get(server_idx, deque()))[-40:]
            if tail_lines:
                tail_text = "\n".join(tail_lines)
                last_error = (
                    f"{last_error}\nRecent server-{server_idx + 1} logs:\n{tail_text}"
                )

        raise RuntimeError(last_error)

    raise RuntimeError(last_error or "PoC request failed with unknown error")


def profile_poc() -> None:
    profile_runs = 10
    dist_threshold = POC_PROFILE_DIST_THRESHOLD
    p_mismatch = POC_PROFILE_P_MISMATCH
    fraud_threshold = POC_PROFILE_FRAUD_THRESHOLD

    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    seq_len = 1024
    k_dim = 12
    tp_size = 1
    api_server_count = 1
    max_model_len = 2048

    public_key = PUBLIC_KEY
    block_hash = BLOCK_HASH

    tp_size, api_server_count, device_slices = _resolve_topology(
        tp_size=tp_size,
        api_server_count=api_server_count,
    )

    ports = [BASE_PORT + i for i in range(api_server_count)]
    base_urls = [f"http://127.0.0.1:{port}" for port in ports]
    _BASE_URL_TO_SERVER_IDX.clear()
    _BASE_URL_TO_SERVER_IDX.update({base_urls[i]: i for i in range(api_server_count)})

    print("=" * 70)
    print("Profiling PoC via OpenAI API /api/v1/pow/generate")
    print(f"Model: {model}")
    print(f"Profile runs: {profile_runs}")
    print(f"TP size: {tp_size}")
    print(f"API servers: {api_server_count}")
    visible_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Visible CUDA devices: {visible_cuda_devices}")
    print(f"Device slices: {device_slices}")
    print(f"max_model_len: {max_model_len}")
    print("model_args:")
    print("  --max-model-len 240000")
    print("  --enable-auto-tool-choice")
    print("  --tool-call-parser hermes")
    print("=" * 70)
    print(f"Using vllm from: {vllm.__file__}")
    print(f"Project root: {PROJECT_ROOT}")
    thresholds_line = (
        f"  dist_threshold={dist_threshold}, "
        f"p_mismatch={p_mismatch}, "
        f"fraud_threshold={fraud_threshold}"
    )
    print(thresholds_line)

    start_delays = [0, 5]

    logs_dir = Path("logs/profile_poc")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [logs_dir / f"server_{i + 1}.log" for i in range(api_server_count)]

    with contextlib.ExitStack() as exit_stack:
        log_files = [
            exit_stack.enter_context(
                log_paths[i].open("w", buffering=1, encoding="utf-8")
            )
            for i in range(api_server_count)
        ]

        server_procs: list[subprocess.Popen] = []
        try:
            print("\nStarting OpenAI API servers in parallel...")
            start_t0 = time.time()
            with ThreadPoolExecutor(max_workers=api_server_count) as executor:
                futures = [
                    executor.submit(
                        _start_server,
                        server_idx,
                        model,
                        tp_size,
                        device_slices[server_idx],
                        ports[server_idx],
                        max_model_len,
                        log_paths[server_idx],
                        log_files[server_idx],
                        (
                            start_delays[server_idx]
                            if server_idx < len(start_delays)
                            else 0
                        ),
                    )
                    for server_idx in range(api_server_count)
                ]

                ready: list[subprocess.Popen | None] = [None] * api_server_count
                for future in as_completed(futures):
                    try:
                        server_idx, proc, log_path = future.result()
                    except Exception as exc:
                        print(f"\n[ERROR] Server startup failed: {exc}")
                        for idx in range(api_server_count):
                            diag_path = (
                                Path("logs/profile_poc") / f"server_{idx + 1}.log"
                            )
                            if diag_path.exists():
                                print(
                                    "\n[LOG DIAG] server "
                                    f"{idx + 1} root-cause excerpt ({diag_path}):\n"
                                    f"{_extract_root_cause(diag_path)}"
                                )
                        raise

                    ready[server_idx] = proc
                    print(
                        f"  API server ready: {server_idx + 1}/{api_server_count} "
                        f"(port={ports[server_idx]}, log={log_path})"
                    )

            for proc in ready:
                if proc is None:
                    raise RuntimeError("Server startup failed")
                server_procs.append(proc)

            print(f"Parallel server startup finished in {time.time() - start_t0:.1f}s")

            batch_size = 16
            print(f"\nRunning {profile_runs} batches with batch_size={batch_size}...")

            warmup_nonces = list(range(batch_size))
            print("\nWarmup run (parallel for all API servers)...")
            with ThreadPoolExecutor(max_workers=api_server_count) as executor:
                futures = [
                    executor.submit(
                        _run_forward_api,
                        base_urls[i],
                        model,
                        block_hash,
                        public_key,
                        warmup_nonces,
                        seq_len,
                        k_dim,
                        batch_size,
                    )
                    for i in range(api_server_count)
                ]
                for future in as_completed(futures):
                    future.result()

            times = []
            combined_times = []
            all_hashes = []
            total_nonces = 0
            per_engine_stats: dict[int, dict[str, float]] = {
                i: {"runs": 0, "nonces": 0, "time": 0.0}
                for i in range(api_server_count)
            }

            print("\nProfiling...")
            validation_payload = _load_validation_payload()
            validation_map = (
                _build_validation_map(validation_payload) if validation_payload else {}
            )
            validated_engines = set()

            for run in range(profile_runs):
                run_start_nonce = run * batch_size * api_server_count
                per_engine_nonces = {
                    engine_idx: list(
                        range(
                            run_start_nonce + engine_idx * batch_size,
                            run_start_nonce + (engine_idx + 1) * batch_size,
                        )
                    )
                    for engine_idx in range(api_server_count)
                }

                wall_t0 = time.time()
                with ThreadPoolExecutor(max_workers=api_server_count) as executor:
                    futures = {
                        executor.submit(
                            _run_forward_api,
                            base_urls[engine_idx],
                            model,
                            block_hash,
                            public_key,
                            per_engine_nonces[engine_idx],
                            seq_len,
                            k_dim,
                            batch_size,
                        ): engine_idx
                        for engine_idx in range(api_server_count)
                    }

                    run_results: dict[
                        int,
                        tuple[float, list[dict[str, Any]], list[int]],
                    ] = {}
                    for future in as_completed(futures):
                        engine_idx = futures[future]
                        elapsed, artifacts = future.result()
                        run_results[engine_idx] = (
                            elapsed,
                            artifacts,
                            per_engine_nonces[engine_idx],
                        )

                wall_elapsed = time.time() - wall_t0
                combined_times.append(wall_elapsed)

                run_total_nonces = 0
                print(f"  Parallel run {run + 1:2d}: wall={wall_elapsed * 1000:.1f}ms")
                for engine_idx in sorted(run_results.keys()):
                    elapsed, artifacts, batch_nonces = run_results[engine_idx]
                    times.append(elapsed)

                    vectors_b64 = [artifact["vector_b64"] for artifact in artifacts]
                    if vectors_b64:
                        all_hashes.append(vectors_b64[0])

                    len_nonces = len(vectors_b64)
                    run_total_nonces += len_nonces
                    total_nonces += len_nonces

                    per_engine_stats[engine_idx]["runs"] += 1
                    per_engine_stats[engine_idx]["nonces"] += len_nonces
                    per_engine_stats[engine_idx]["time"] += elapsed

                    nonces_per_sec = len_nonces / elapsed if elapsed > 0 else 0
                    ms_per_nonce = (
                        (elapsed * 1000 / len_nonces) if len_nonces > 0 else 0
                    )
                    print(
                        f"    Engine {engine_idx + 1}: {elapsed * 1000:.1f}ms, "
                        f"{len_nonces} nonces ({nonces_per_sec:.1f}/sec, "
                        f"{ms_per_nonce:.2f}ms/nonce)"
                    )

                    if validation_map and engine_idx not in validated_engines:
                        computed_artifacts = artifacts

                        print(f"\n[DEBUG] Validation Info (engine {engine_idx + 1}):")

                        validation_keys = sorted(validation_map)
                        print(
                            "  validation_map has "
                            f"{len(validation_map)} entries: {validation_keys}"
                        )
                        print(
                            "  batch_nonces: "
                            f"{batch_nonces[:5]}... (first 5 of {len(batch_nonces)})"
                        )
                        print(
                            "  computed_artifacts has "
                            f"{len(computed_artifacts)} entries"
                        )

                        matching_nonces = [
                            n for n in batch_nonces if n in validation_map
                        ]
                        print(
                            "  matching nonces: "
                            f"{matching_nonces} ({len(matching_nonces)} found)"
                        )

                        if matching_nonces:
                            for nonce in matching_nonces:
                                expected = validation_map[nonce]
                                computed = next(
                                    a["vector_b64"]
                                    for a in computed_artifacts
                                    if a["nonce"] == nonce
                                )
                                match = "✓" if expected == computed else "✗"
                                print(f"    nonce {nonce}: {match}")
                                if expected != computed:
                                    print(f"      expected: {expected}")
                                    print(f"      got:      {computed}")

                        try:
                            validation_result = _run_validation(
                                computed_artifacts,
                                validation_map,
                                dist_threshold=dist_threshold,
                                p_mismatch=p_mismatch,
                                fraud_threshold=fraud_threshold,
                                k_dim=k_dim,
                            )
                        except Exception as exc:
                            print(
                                "\n[WARN] Validation failed "
                                f"({type(exc).__name__}): {exc}"
                            )
                            validation_result = {
                                "n_total": len(computed_artifacts),
                                "n_mismatch": len(computed_artifacts),
                                "p_value": 0.0,
                                "fraud_detected": True,
                                "mismatch_nonces": [],
                            }

                        validated_engines.add(engine_idx)
                        print(f"\nValidation result (engine {engine_idx + 1}):")
                        print(
                            f"  n_total={validation_result['n_total']}, "
                            f"n_mismatch={validation_result['n_mismatch']}, "
                            f"p_value={validation_result['p_value']:.6f}, "
                            f"fraud_detected={validation_result['fraud_detected']}"
                        )
                        if validation_result.get("mismatch_nonces"):
                            print(
                                "  mismatch_nonces="
                                f"{validation_result['mismatch_nonces']}"
                            )
                        print()

                combined_rate = (
                    run_total_nonces / wall_elapsed if wall_elapsed > 0 else 0
                )
                print(
                    f"    Combined: {run_total_nonces} nonces in "
                    f"{wall_elapsed * 1000:.1f}ms ({combined_rate:.1f}/sec)"
                )

            print("\n" + "=" * 70)
            print("RESULTS:")
            print("=" * 70)

            if times:
                avg_time = sum(times) / len(times)
                avg_parallel_time = (
                    (sum(combined_times) / len(combined_times))
                    if combined_times
                    else avg_time
                )
                avg_nonces = total_nonces / len(times)
                avg_rate = avg_nonces / avg_time if avg_time > 0 else 0
                avg_combined_nonces_per_run = (
                    (total_nonces / len(combined_times))
                    if combined_times
                    else avg_nonces
                )
                avg_combined_rate = (
                    (avg_combined_nonces_per_run / avg_parallel_time)
                    if avg_parallel_time > 0
                    else 0
                )
                time_per_nonce = avg_time / avg_nonces if avg_nonces > 0 else 0

                print(f"Batch size used: {batch_size}")
                print(f"Total batches: {len(times)}")
                print(f"Parallel rounds: {len(combined_times)}")
                print(f"Total nonces: {total_nonces}")
                print(f"Average batch time: {avg_time * 1000:.1f}ms")
                print(f"Average parallel round time: {avg_parallel_time * 1000:.1f}ms")
                print(f"Average rate: {avg_rate:.2f} nonces/sec")
                print(
                    "Average combined parallel rate: "
                    f"{avg_combined_rate:.2f} nonces/sec"
                )
                print(f"Average rate: {avg_rate * 60:.0f} nonces/min")
                print(f"Time per nonce: {time_per_nonce * 1000:.2f}ms")

                print("\nPer-engine summary:")
                for engine_idx in range(api_server_count):
                    e_runs = per_engine_stats[engine_idx]["runs"]
                    e_nonces = per_engine_stats[engine_idx]["nonces"]
                    e_time = per_engine_stats[engine_idx]["time"]
                    e_rate = e_nonces / e_time if e_time > 0 else 0
                    e_ms_per_nonce = (e_time * 1000 / e_nonces) if e_nonces > 0 else 0
                    print(
                        f"  Engine {engine_idx + 1}: runs={int(e_runs)}, "
                        f"nonces={int(e_nonces)}, rate={e_rate:.2f}/sec, "
                        f"ms/nonce={e_ms_per_nonce:.2f}"
                    )

                if len(times) > 1:
                    min_time = min(times) * 1000
                    max_time = max(times) * 1000
                    variance = (
                        ((max_time - min_time) / (avg_time * 1000) * 100)
                        if avg_time > 0
                        else 0
                    )
                    print(
                        "\nBatch time variance: "
                        f"{min_time:.1f} - {max_time:.1f}ms (±{variance:.1f}%)"
                    )

                for i, h in enumerate(set(all_hashes)):
                    print(f"  Variant {i + 1}: {h}")

                target_ms_per_nonce = 57
                current_ms = time_per_nonce * 1000
                gap = (current_ms - target_ms_per_nonce) / target_ms_per_nonce * 100

                print(f"\n{'=' * 70}")
                print("Performance vs target:")
                print(f"  Current: {current_ms:.2f}ms/nonce")
                print(f"  Target:  {target_ms_per_nonce:.2f}ms/nonce")
                if gap > 0:
                    print(f"  Gap: {gap:.1f}% slower (need {gap:.0f}% improvement)")
                else:
                    print(f"  ✓ Exceeded target by {-gap:.1f}%!")

                if avg_rate > 0:
                    est_1k_sec = 1000 / avg_rate
                    est_10k_sec = 10000 / avg_rate
                    print(
                        "\nEstimated time for 1000 nonces: "
                        f"{est_1k_sec:.1f}s ({est_1k_sec / 60:.1f}min)"
                    )
                    print(
                        "Estimated time for 10000 nonces: "
                        f"{est_10k_sec:.1f}s ({est_10k_sec / 60:.1f}min)"
                    )
            else:
                print("No timing data collected!")
        finally:
            for proc in server_procs:
                _stop_process(proc)


if __name__ == "__main__":
    profile_poc()
