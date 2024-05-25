# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ks.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x08ks.proto\x12\x0ckuaishouPubf\"\xec\x01\n\x0e\x43SWebEnterRoom\x12\x13\n\x0bpayloadType\x18\x01 \x01(\x03\x12\x35\n\x07payload\x18\x03 \x01(\x0b\x32$.kuaishouPubf.CSWebEnterRoom.Payload\x1a\x8d\x01\n\x07Payload\x12\r\n\x05token\x18\x01 \x01(\t\x12\x14\n\x0cliveStreamId\x18\x02 \x01(\t\x12\x16\n\x0ereconnectCount\x18\x03 \x01(\r\x12\x15\n\rlastErrorCode\x18\x04 \x01(\r\x12\x0e\n\x06\x65xpTag\x18\x05 \x01(\t\x12\x0e\n\x06\x61ttach\x18\x06 \x01(\t\x12\x0e\n\x06pageId\x18\x07 \x01(\t\"`\n\x11SCWebEnterRoomAck\x12\x16\n\x0eminReconnectMs\x18\x01 \x01(\x04\x12\x16\n\x0emaxReconnectMs\x18\x02 \x01(\x04\x12\x1b\n\x13heartbeatIntervalMs\x18\x03 \x01(\x04\"H\n\x0eSimpleUserInfo\x12\x13\n\x0bprincipalId\x18\x01 \x01(\t\x12\x10\n\x08userName\x18\x02 \x01(\t\x12\x0f\n\x07headUrl\x18\x03 \x01(\t\"\xb7\x01\n\x13WebWatchingUserInfo\x12*\n\x04user\x18\x01 \x01(\x0b\x32\x1c.kuaishouPubf.SimpleUserInfo\x12\x0f\n\x07offline\x18\x02 \x01(\x08\x12\r\n\x05tuhao\x18\x03 \x01(\x08\x12=\n\x11liveAssistantType\x18\x04 \x01(\x0e\x32\".kuaishouPubf.WebLiveAssistantType\x12\x15\n\rdisplayKsCoin\x18\x05 \x01(\t\"\x88\x01\n\x16SCWebLiveWatchingUsers\x12\x37\n\x0cwatchingUser\x18\x01 \x03(\x0b\x32!.kuaishouPubf.WebWatchingUserInfo\x12\x1c\n\x14\x64isplayWatchingCount\x18\x02 \x01(\t\x12\x17\n\x0fpendingDuration\x18\x03 \x01(\x04\"z\n\x0e\x43SWebHeartbeat\x12\x13\n\x0bpayloadType\x18\x01 \x01(\x03\x12\x35\n\x07payload\x18\x03 \x01(\x0b\x32$.kuaishouPubf.CSWebHeartbeat.Payload\x1a\x1c\n\x07Payload\x12\x11\n\ttimestamp\x18\x01 \x01(\x04\"\x88\x01\n\rSocketMessage\x12.\n\x0bpayloadType\x18\x01 \x01(\x0e\x32\x19.kuaishouPubf.PayloadType\x12\x36\n\x0f\x63ompressionType\x18\x02 \x01(\x0e\x32\x1d.kuaishouPubf.CompressionType\x12\x0f\n\x07payload\x18\x03 \x01(\x0c\"<\n\x0eSCHeartbeatAck\x12\x11\n\ttimestamp\x18\x01 \x01(\x04\x12\x17\n\x0f\x63lientTimestamp\x18\x02 \x01(\x04\"\xfc\x01\n\x0eWebCommentFeed\x12\n\n\x02id\x18\x01 \x01(\t\x12*\n\x04user\x18\x02 \x01(\x0b\x32\x1c.kuaishouPubf.SimpleUserInfo\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\t\x12\x12\n\ndeviceHash\x18\x04 \x01(\t\x12\x10\n\x08sortRank\x18\x05 \x01(\x04\x12\r\n\x05\x63olor\x18\x06 \x01(\t\x12\x36\n\x08showType\x18\x07 \x01(\x0e\x32$.kuaishouPubf.WebCommentFeedShowType\x12\x34\n\x0bsenderState\x18\x08 \x01(\x0b\x32\x1f.kuaishouPubf.LiveAudienceState\"\xa3\x02\n\x11LiveAudienceState\x12\x15\n\risFromFansTop\x18\x01 \x01(\x08\x12\r\n\x05isKoi\x18\x02 \x01(\x08\x12\x32\n\rassistantType\x18\x03 \x01(\x0e\x32\x1b.kuaishouPubf.AssistantType\x12\x1e\n\x16\x66\x61nsGroupIntimacyLevel\x18\x04 \x01(\r\x12/\n\tnameplate\x18\x05 \x01(\x0b\x32\x1c.kuaishouPubf.GzoneNameplate\x12<\n\x12liveFansGroupState\x18\x06 \x01(\x0b\x32 .kuaishouPubf.LiveFansGroupState\x12\x13\n\x0bwealthGrade\x18\x07 \x01(\r\x12\x10\n\x08\x62\x61\x64geKey\x18\x08 \x01(\t\"K\n\x12LiveFansGroupState\x12\x15\n\rintimacyLevel\x18\x01 \x01(\r\x12\x1e\n\x16\x65nterRoomSpecialEffect\x18\x02 \x01(\r\"N\n\x0eGzoneNameplate\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\"\n\x04urls\x18\x03 \x03(\x0b\x32\x14.kuaishouPubf.PicUrl\"B\n\x06PicUrl\x12\x0b\n\x03\x63\x64n\x18\x01 \x01(\t\x12\x0b\n\x03url\x18\x02 \x01(\t\x12\x12\n\nurlPattern\x18\x03 \x01(\t\x12\n\n\x02ip\x18\x04 \x01(\t\"\xdd\x03\n\rSCWebFeedPush\x12\x1c\n\x14\x64isplayWatchingCount\x18\x01 \x01(\t\x12\x18\n\x10\x64isplayLikeCount\x18\x02 \x01(\t\x12\x18\n\x10pendingLikeCount\x18\x03 \x01(\x04\x12\x14\n\x0cpushInterval\x18\x04 \x01(\x04\x12\x32\n\x0c\x63ommentFeeds\x18\x05 \x03(\x0b\x32\x1c.kuaishouPubf.WebCommentFeed\x12\x15\n\rcommentCursor\x18\x06 \x01(\t\x12;\n\x10\x63omboCommentFeed\x18\x07 \x03(\x0b\x32!.kuaishouPubf.WebComboCommentFeed\x12,\n\tlikeFeeds\x18\x08 \x03(\x0b\x32\x19.kuaishouPubf.WebLikeFeed\x12,\n\tgiftFeeds\x18\t \x03(\x0b\x32\x19.kuaishouPubf.WebGiftFeed\x12\x12\n\ngiftCursor\x18\n \x01(\t\x12<\n\x11systemNoticeFeeds\x18\x0b \x03(\x0b\x32!.kuaishouPubf.WebSystemNoticeFeed\x12.\n\nshareFeeds\x18\x0c \x03(\x0b\x32\x1a.kuaishouPubf.WebShareFeed\"\xd5\x01\n\x0cWebShareFeed\x12\n\n\x02id\x18\x01 \x01(\t\x12*\n\x04user\x18\x02 \x01(\x0b\x32\x1c.kuaishouPubf.SimpleUserInfo\x12\x0c\n\x04time\x18\x03 \x01(\x04\x12\x1a\n\x12thirdPartyPlatform\x18\x04 \x01(\r\x12\x10\n\x08sortRank\x18\x05 \x01(\x04\x12=\n\x11liveAssistantType\x18\x06 \x01(\x0e\x32\".kuaishouPubf.WebLiveAssistantType\x12\x12\n\ndeviceHash\x18\x07 \x01(\t\"\xc7\x01\n\x13WebSystemNoticeFeed\x12\n\n\x02id\x18\x01 \x01(\t\x12*\n\x04user\x18\x02 \x01(\x0b\x32\x1c.kuaishouPubf.SimpleUserInfo\x12\x0c\n\x04time\x18\x03 \x01(\x04\x12\x0f\n\x07\x63ontent\x18\x04 \x01(\t\x12\x17\n\x0f\x64isplayDuration\x18\x05 \x01(\x04\x12\x10\n\x08sortRank\x18\x06 \x01(\x04\x12.\n\x0b\x64isplayType\x18\x07 \x01(\x0e\x32\x19.kuaishouPubf.DisplayType\"\xb4\x03\n\x0bWebGiftFeed\x12\n\n\x02id\x18\x01 \x01(\t\x12*\n\x04user\x18\x02 \x01(\x0b\x32\x1c.kuaishouPubf.SimpleUserInfo\x12\x0c\n\x04time\x18\x03 \x01(\x04\x12\x0e\n\x06giftId\x18\x04 \x01(\r\x12\x10\n\x08sortRank\x18\x05 \x01(\x04\x12\x10\n\x08mergeKey\x18\x06 \x01(\t\x12\x11\n\tbatchSize\x18\x07 \x01(\r\x12\x12\n\ncomboCount\x18\x08 \x01(\r\x12\x0c\n\x04rank\x18\t \x01(\r\x12\x16\n\x0e\x65xpireDuration\x18\n \x01(\x04\x12\x17\n\x0f\x63lientTimestamp\x18\x0b \x01(\x04\x12\x1b\n\x13slotDisplayDuration\x18\x0c \x01(\x04\x12\x11\n\tstarLevel\x18\r \x01(\r\x12*\n\tstyleType\x18\x0e \x01(\x0e\x32\x17.kuaishouPubf.StyleType\x12=\n\x11liveAssistantType\x18\x0f \x01(\x0e\x32\".kuaishouPubf.WebLiveAssistantType\x12\x12\n\ndeviceHash\x18\x10 \x01(\t\x12\x16\n\x0e\x64\x61nmakuDisplay\x18\x11 \x01(\x08\"k\n\x0bWebLikeFeed\x12\n\n\x02id\x18\x01 \x01(\t\x12*\n\x04user\x18\x02 \x01(\x0b\x32\x1c.kuaishouPubf.SimpleUserInfo\x12\x10\n\x08sortRank\x18\x03 \x01(\x04\x12\x12\n\ndeviceHash\x18\x04 \x01(\t\"F\n\x13WebComboCommentFeed\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\t\x12\x12\n\ncomboCount\x18\x03 \x01(\r*L\n\x0f\x43ompressionType\x12\x1c\n\x18\x43OMPRESSION_TYPE_UNKNOWN\x10\x00\x12\x08\n\x04NONE\x10\x01\x12\x08\n\x04GZIP\x10\x02\x12\x07\n\x03\x41\x45S\x10\x03*\x98\x0b\n\x0bPayloadType\x12\x18\n\x14PAYLOAD_TYPE_UNKNOWN\x10\x00\x12\x10\n\x0c\x43S_HEARTBEAT\x10\x01\x12\x0c\n\x08\x43S_ERROR\x10\x03\x12\x0b\n\x07\x43S_PING\x10\x04\x12\x10\n\x0cPS_HOST_INFO\x10\x33\x12\x14\n\x10SC_HEARTBEAT_ACK\x10\x65\x12\x0b\n\x07SC_ECHO\x10\x66\x12\x0c\n\x08SC_ERROR\x10g\x12\x0f\n\x0bSC_PING_ACK\x10h\x12\x0b\n\x07SC_INFO\x10i\x12\x12\n\rCS_ENTER_ROOM\x10\xc8\x01\x12\x12\n\rCS_USER_PAUSE\x10\xc9\x01\x12\x11\n\x0c\x43S_USER_EXIT\x10\xca\x01\x12 \n\x1b\x43S_AUTHOR_PUSH_TRAFFIC_ZERO\x10\xcb\x01\x12\x14\n\x0f\x43S_HORSE_RACING\x10\xcc\x01\x12\x11\n\x0c\x43S_RACE_LOSE\x10\xcd\x01\x12\x13\n\x0e\x43S_VOIP_SIGNAL\x10\xce\x01\x12\x16\n\x11SC_ENTER_ROOM_ACK\x10\xac\x02\x12\x14\n\x0fSC_AUTHOR_PAUSE\x10\xad\x02\x12\x15\n\x10SC_AUTHOR_RESUME\x10\xae\x02\x12 \n\x1bSC_AUTHOR_PUSH_TRAFFIC_ZERO\x10\xaf\x02\x12\x1d\n\x18SC_AUTHOR_HEARTBEAT_MISS\x10\xb0\x02\x12\x13\n\x0eSC_PIP_STARTED\x10\xb1\x02\x12\x11\n\x0cSC_PIP_ENDED\x10\xb2\x02\x12\x18\n\x13SC_HORSE_RACING_ACK\x10\xb3\x02\x12\x13\n\x0eSC_VOIP_SIGNAL\x10\xb4\x02\x12\x11\n\x0cSC_FEED_PUSH\x10\xb6\x02\x12\x18\n\x13SC_ASSISTANT_STATUS\x10\xb7\x02\x12\x16\n\x11SC_REFRESH_WALLET\x10\xb8\x02\x12\x16\n\x11SC_LIVE_CHAT_CALL\x10\xc0\x02\x12\x1f\n\x1aSC_LIVE_CHAT_CALL_ACCEPTED\x10\xc1\x02\x12\x1f\n\x1aSC_LIVE_CHAT_CALL_REJECTED\x10\xc2\x02\x12\x17\n\x12SC_LIVE_CHAT_READY\x10\xc3\x02\x12\x1b\n\x16SC_LIVE_CHAT_GUEST_END\x10\xc4\x02\x12\x17\n\x12SC_LIVE_CHAT_ENDED\x10\xc5\x02\x12$\n\x1fSC_RENDERING_MAGIC_FACE_DISABLE\x10\xc6\x02\x12#\n\x1eSC_RENDERING_MAGIC_FACE_ENABLE\x10\xc7\x02\x12\x15\n\x10SC_RED_PACK_FEED\x10\xca\x02\x12\x1a\n\x15SC_LIVE_WATCHING_LIST\x10\xd4\x02\x12 \n\x1bSC_LIVE_QUIZ_QUESTION_ASKED\x10\xde\x02\x12#\n\x1eSC_LIVE_QUIZ_QUESTION_REVIEWED\x10\xdf\x02\x12\x16\n\x11SC_LIVE_QUIZ_SYNC\x10\xe0\x02\x12\x17\n\x12SC_LIVE_QUIZ_ENDED\x10\xe1\x02\x12\x19\n\x14SC_LIVE_QUIZ_WINNERS\x10\xe2\x02\x12\x1b\n\x16SC_SUSPECTED_VIOLATION\x10\xe3\x02\x12\x13\n\x0eSC_SHOP_OPENED\x10\xe8\x02\x12\x13\n\x0eSC_SHOP_CLOSED\x10\xe9\x02\x12\x14\n\x0fSC_GUESS_OPENED\x10\xf2\x02\x12\x14\n\x0fSC_GUESS_CLOSED\x10\xf3\x02\x12\x15\n\x10SC_PK_INVITATION\x10\xfc\x02\x12\x14\n\x0fSC_PK_STATISTIC\x10\xfd\x02\x12\x15\n\x10SC_RIDDLE_OPENED\x10\x86\x03\x12\x16\n\x11SC_RIDDLE_CLOESED\x10\x87\x03\x12\x14\n\x0fSC_RIDE_CHANGED\x10\x9c\x03\x12\x13\n\x0eSC_BET_CHANGED\x10\xb9\x03\x12\x12\n\rSC_BET_CLOSED\x10\xba\x03\x12)\n$SC_LIVE_SPECIAL_ACCOUNT_CONFIG_STATE\x10\x85\x05\x12\x31\n,SC_LIVE_WARNING_MASK_STATUS_CHANGED_AUDIENCE\x10\xf6\x05*a\n\x14WebLiveAssistantType\x12\x32\n.WEB_LIVE_ASSISTANT_TYPE_UNKNOWN_ASSISTANT_TYPE\x10\x00\x12\t\n\x05SUPER\x10\x01\x12\n\n\x06JUNIOR\x10\x02*V\n\x16WebCommentFeedShowType\x12\x15\n\x11\x46\x45\x45\x44_SHOW_UNKNOWN\x10\x00\x12\x14\n\x10\x46\x45\x45\x44_SHOW_NORMAL\x10\x01\x12\x0f\n\x0b\x46\x45\x45\x44_HIDDEN\x10\x02*V\n\rAssistantType\x12\x1a\n\x16UNKNOWN_ASSISTANT_TYPE\x10\x00\x12\x13\n\x0f\x41SSISTANT_SUPER\x10\x01\x12\x14\n\x10\x41SSISTANT_JUNIOR\x10\x02*\x9c\x01\n\tStyleType\x12\x11\n\rUNKNOWN_STYLE\x10\x00\x12\x10\n\x0c\x42\x41TCH_STAR_0\x10\x01\x12\x10\n\x0c\x42\x41TCH_STAR_1\x10\x02\x12\x10\n\x0c\x42\x41TCH_STAR_2\x10\x03\x12\x10\n\x0c\x42\x41TCH_STAR_3\x10\x04\x12\x10\n\x0c\x42\x41TCH_STAR_4\x10\x05\x12\x10\n\x0c\x42\x41TCH_STAR_5\x10\x06\x12\x10\n\x0c\x42\x41TCH_STAR_6\x10\x07*J\n\x0b\x44isplayType\x12\x18\n\x14UNKNOWN_DISPLAY_TYPE\x10\x00\x12\x0b\n\x07\x43OMMENT\x10\x01\x12\t\n\x05\x41LERT\x10\x02\x12\t\n\x05TOAST\x10\x03\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ks_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_COMPRESSIONTYPE']._serialized_start=3379
  _globals['_COMPRESSIONTYPE']._serialized_end=3455
  _globals['_PAYLOADTYPE']._serialized_start=3458
  _globals['_PAYLOADTYPE']._serialized_end=4890
  _globals['_WEBLIVEASSISTANTTYPE']._serialized_start=4892
  _globals['_WEBLIVEASSISTANTTYPE']._serialized_end=4989
  _globals['_WEBCOMMENTFEEDSHOWTYPE']._serialized_start=4991
  _globals['_WEBCOMMENTFEEDSHOWTYPE']._serialized_end=5077
  _globals['_ASSISTANTTYPE']._serialized_start=5079
  _globals['_ASSISTANTTYPE']._serialized_end=5165
  _globals['_STYLETYPE']._serialized_start=5168
  _globals['_STYLETYPE']._serialized_end=5324
  _globals['_DISPLAYTYPE']._serialized_start=5326
  _globals['_DISPLAYTYPE']._serialized_end=5400
  _globals['_CSWEBENTERROOM']._serialized_start=27
  _globals['_CSWEBENTERROOM']._serialized_end=263
  _globals['_CSWEBENTERROOM_PAYLOAD']._serialized_start=122
  _globals['_CSWEBENTERROOM_PAYLOAD']._serialized_end=263
  _globals['_SCWEBENTERROOMACK']._serialized_start=265
  _globals['_SCWEBENTERROOMACK']._serialized_end=361
  _globals['_SIMPLEUSERINFO']._serialized_start=363
  _globals['_SIMPLEUSERINFO']._serialized_end=435
  _globals['_WEBWATCHINGUSERINFO']._serialized_start=438
  _globals['_WEBWATCHINGUSERINFO']._serialized_end=621
  _globals['_SCWEBLIVEWATCHINGUSERS']._serialized_start=624
  _globals['_SCWEBLIVEWATCHINGUSERS']._serialized_end=760
  _globals['_CSWEBHEARTBEAT']._serialized_start=762
  _globals['_CSWEBHEARTBEAT']._serialized_end=884
  _globals['_CSWEBHEARTBEAT_PAYLOAD']._serialized_start=856
  _globals['_CSWEBHEARTBEAT_PAYLOAD']._serialized_end=884
  _globals['_SOCKETMESSAGE']._serialized_start=887
  _globals['_SOCKETMESSAGE']._serialized_end=1023
  _globals['_SCHEARTBEATACK']._serialized_start=1025
  _globals['_SCHEARTBEATACK']._serialized_end=1085
  _globals['_WEBCOMMENTFEED']._serialized_start=1088
  _globals['_WEBCOMMENTFEED']._serialized_end=1340
  _globals['_LIVEAUDIENCESTATE']._serialized_start=1343
  _globals['_LIVEAUDIENCESTATE']._serialized_end=1634
  _globals['_LIVEFANSGROUPSTATE']._serialized_start=1636
  _globals['_LIVEFANSGROUPSTATE']._serialized_end=1711
  _globals['_GZONENAMEPLATE']._serialized_start=1713
  _globals['_GZONENAMEPLATE']._serialized_end=1791
  _globals['_PICURL']._serialized_start=1793
  _globals['_PICURL']._serialized_end=1859
  _globals['_SCWEBFEEDPUSH']._serialized_start=1862
  _globals['_SCWEBFEEDPUSH']._serialized_end=2339
  _globals['_WEBSHAREFEED']._serialized_start=2342
  _globals['_WEBSHAREFEED']._serialized_end=2555
  _globals['_WEBSYSTEMNOTICEFEED']._serialized_start=2558
  _globals['_WEBSYSTEMNOTICEFEED']._serialized_end=2757
  _globals['_WEBGIFTFEED']._serialized_start=2760
  _globals['_WEBGIFTFEED']._serialized_end=3196
  _globals['_WEBLIKEFEED']._serialized_start=3198
  _globals['_WEBLIKEFEED']._serialized_end=3305
  _globals['_WEBCOMBOCOMMENTFEED']._serialized_start=3307
  _globals['_WEBCOMBOCOMMENTFEED']._serialized_end=3377
# @@protoc_insertion_point(module_scope)
