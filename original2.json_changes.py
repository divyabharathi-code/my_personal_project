import base64
import json

test_event = {
    "objects": [
        {
            "eventType": "UPDATE",
            "entitlements": [],
            "changes": [],
            "objectId": "5858841",
            "objectType": "GEO_PRODUCT_OFFERING",
            "fullObject": '{"statInd":"A","gpoId":5858841,"sizeProdId":3350678,"primMktgTypeId":1,"crtTmst":"2017-08-22T19:32:58.000Z","gpoNotes":null,"frstOfrDt":"2018-07-01","statChngTmst":"2017-08-22T19:32:58.000Z","gmoId":8165116,"gpoSpnsrCaovStatInd":null,"mdfdByAcctCd":"CHG3322416_LAUNCH","productOffering":{"statInd":"A","primMktgTypeId":1,"product":{"prodCd":"864349-406","gblCatSumCd":"1007","siloId":null,"gblCatCoreFocsCd":"2012","saleSzRng":"4-12,13,14","colorway":{"primColrId":628,"clrwyId":796156,"extrnFullColrDesc":"UNIV BLUE/LT BONE-HABANERO RED","verNbr":2},"prodId":3350678,"subCatCd":"005104","catCd":"000491","teamId":null,"prodClrwyCd":"406","style":{"stylNbr":"864349","fnshdGdsInd":true,"mrchClsfnId":null,"stylNm":"NIKE SB ZOOM BLAZER MID","verNbr":873},"verNbr":16,"fitCd":"GF"},"prodCrtnInittrId":1,"aaInd":false,"verNbr":167,"poId":5238781,"erlstAllowdOfrDt":"2018-07-01"},"chnlSgmtnId":[147,153,16,28],"netDpCpoRlupQty":696,"netBkgsCpoRlupQty":651,"saleSzRng":"6-12, 13, 14","launch":{"lnchCd":"D","lnchDt":"2018-07-10"},"typeGrpCd":"10","cycleYear":{"cyclYrAbbrn":"FA18","yrLongDesc":2018},"crtdByAcctCd":"LJOYC1","size":{"saleSzDesc":["6","6.5","7","7.5","8","8.5","9","9.5","10","10.5","11","11.5","12","13","14"]},"mktgTypeCmnt":null,"regId":76,"rpInd":null,"gpoMrchFcstQty":0,"verNbr":59,"chngTmst":"2025-02-14T04:09:47.000Z","gpoCaovStatInd":"NEW"}',
            "sourceSystem": "CPS_ADAPTOR",
            "kmsKey": None,
            "apiVersion": 1,
            "correlationId": None,
            "domain": "LM",
            "objectVersion": 59,
            "modifiedBy": None,
            "modifiedTimestamp": None,
        }
    ]
}

a = base64.b64encode(
                    json.dumps(test_event).encode("utf-8")
                )

print(a)
test_events = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": '""',
                "sequenceNumber": "49659357891078650911779273200601434544624828992096043074",
                "data": "eyJvYmplY3RzIjogW3siZXZlbnRUeXBlIjogIlVQREFURSIsICJlbnRpdGxlbWVudHMiOiBbXSwgImNoYW5nZXMiOiBbXSwgIm9iamVjdElkIjogIjU4NTg4NDEiLCAib2JqZWN0VHlwZSI6ICJHRU9fUFJPRFVDVF9PRkZFUklORyIsICJmdWxsT2JqZWN0IjogIntcInN0YXRJbmRcIjpcIkFcIixcImdwb0lkXCI6NTg1ODg0MSxcInNpemVQcm9kSWRcIjozMzUwNjc4LFwicHJpbU1rdGdUeXBlSWRcIjoxLFwiY3J0VG1zdFwiOlwiMjAxNy0wOC0yMlQxOTozMjo1OC4wMDBaXCIsXCJncG9Ob3Rlc1wiOm51bGwsXCJmcnN0T2ZyRHRcIjpcIjIwMTgtMDctMDFcIixcInN0YXRDaG5nVG1zdFwiOlwiMjAxNy0wOC0yMlQxOTozMjo1OC4wMDBaXCIsXCJnbW9JZFwiOjgxNjUxMTYsXCJncG9TcG5zckNhb3ZTdGF0SW5kXCI6bnVsbCxcIm1kZmRCeUFjY3RDZFwiOlwiQ0hHMzMyMjQxNl9MQVVOQ0hcIixcInByb2R1Y3RPZmZlcmluZ1wiOntcInN0YXRJbmRcIjpcIkFcIixcInByaW1Na3RnVHlwZUlkXCI6MSxcInByb2R1Y3RcIjp7XCJwcm9kQ2RcIjpcIjg2NDM0OS00MDZcIixcImdibENhdFN1bUNkXCI6XCIxMDA3XCIsXCJzaWxvSWRcIjpudWxsLFwiZ2JsQ2F0Q29yZUZvY3NDZFwiOlwiMjAxMlwiLFwic2FsZVN6Um5nXCI6XCI0LTEyLDEzLDE0XCIsXCJjb2xvcndheVwiOntcInByaW1Db2xySWRcIjo2MjgsXCJjbHJ3eUlkXCI6Nzk2MTU2LFwiZXh0cm5GdWxsQ29sckRlc2NcIjpcIlVOSVYgQkxVRS9MVCBCT05FLUhBQkFORVJPIFJFRFwiLFwidmVyTmJyXCI6Mn0sXCJwcm9kSWRcIjozMzUwNjc4LFwic3ViQ2F0Q2RcIjpcIjAwNTEwNFwiLFwiY2F0Q2RcIjpcIjAwMDQ5MVwiLFwidGVhbUlkXCI6bnVsbCxcInByb2RDbHJ3eUNkXCI6XCI0MDZcIixcInN0eWxlXCI6e1wic3R5bE5iclwiOlwiODY0MzQ5XCIsXCJmbnNoZEdkc0luZFwiOnRydWUsXCJtcmNoQ2xzZm5JZFwiOm51bGwsXCJzdHlsTm1cIjpcIk5JS0UgU0IgWk9PTSBCTEFaRVIgTUlEXCIsXCJ2ZXJOYnJcIjo4NzN9LFwidmVyTmJyXCI6MTYsXCJmaXRDZFwiOlwiR0ZcIn0sXCJwcm9kQ3J0bkluaXR0cklkXCI6MSxcImFhSW5kXCI6ZmFsc2UsXCJ2ZXJOYnJcIjoxNjcsXCJwb0lkXCI6NTIzODc4MSxcImVybHN0QWxsb3dkT2ZyRHRcIjpcIjIwMTgtMDctMDFcIn0sXCJjaG5sU2dtdG5JZFwiOlsxNDcsMTUzLDE2LDI4XSxcIm5ldERwQ3BvUmx1cFF0eVwiOjY5NixcIm5ldEJrZ3NDcG9SbHVwUXR5XCI6NjUxLFwic2FsZVN6Um5nXCI6XCI2LTEyLCAxMywgMTRcIixcImxhdW5jaFwiOntcImxuY2hDZFwiOlwiRFwiLFwibG5jaER0XCI6XCIyMDE4LTA3LTEwXCJ9LFwidHlwZUdycENkXCI6XCIxMFwiLFwiY3ljbGVZZWFyXCI6e1wiY3ljbFlyQWJicm5cIjpcIkZBMThcIixcInlyTG9uZ0Rlc2NcIjoyMDE4fSxcImNydGRCeUFjY3RDZFwiOlwiTEpPWUMxXCIsXCJzaXplXCI6e1wic2FsZVN6RGVzY1wiOltcIjZcIixcIjYuNVwiLFwiN1wiLFwiNy41XCIsXCI4XCIsXCI4LjVcIixcIjlcIixcIjkuNVwiLFwiMTBcIixcIjEwLjVcIixcIjExXCIsXCIxMS41XCIsXCIxMlwiLFwiMTNcIixcIjE0XCJdfSxcIm1rdGdUeXBlQ21udFwiOm51bGwsXCJyZWdJZFwiOjc2LFwicnBJbmRcIjpudWxsLFwiZ3BvTXJjaEZjc3RRdHlcIjowLFwidmVyTmJyXCI6NTksXCJjaG5nVG1zdFwiOlwiMjAyNS0wMi0xNFQwNDowOTo0Ny4wMDBaXCIsXCJncG9DYW92U3RhdEluZFwiOlwiTkVXXCJ9IiwgInNvdXJjZVN5c3RlbSI6ICJDUFNfQURBUFRPUiIsICJrbXNLZXkiOiBudWxsLCAiYXBpVmVyc2lvbiI6IDEsICJjb3JyZWxhdGlvbklkIjogbnVsbCwgImRvbWFpbiI6ICJMTSIsICJvYmplY3RWZXJzaW9uIjogNTksICJtb2RpZmllZEJ5IjogbnVsbCwgIm1vZGlmaWVkVGltZXN0YW1wIjogbnVsbH1dfQ==",
                "approximateArrivalTimestamp": 1736416137.674,
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000004:49659357891078650911779273200601434544624828992096043074",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::169344650340:role/pmp-preprod-cps-service-role-stg",
            "awsRegion": "us-west-2",
            "eventSourceARN": "arn:aws:kinesis:us-west-2:169344650340:stream/pmp-preprod-cps-mmx-qa",
        }
    ]
}
