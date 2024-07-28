/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"




PyDoc_STRVAR(doc_FbxAnimCurveNode_Create, "Create(FbxManager, str) -> FbxAnimCurveNode\n"
    "Create(FbxObject, str) -> FbxAnimCurveNode");

extern "C" {static PyObject *meth_FbxAnimCurveNode_Create(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_Create(PyObject *, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxManager* a0;
        const char* a1;
        PyObject *a1Keep;

        if (sipParseArgs(&sipParseErr, sipArgs, "J8A8", sipType_FbxManager, &a0, &a1Keep, &a1))
        {
             ::FbxAnimCurveNode*sipRes;

            sipRes =  ::FbxAnimCurveNode::Create(a0,a1);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxAnimCurveNode,NULL);
        }
    }

    {
         ::FbxObject* a0;
        const char* a1;
        PyObject *a1Keep;

        if (sipParseArgs(&sipParseErr, sipArgs, "J8A8", sipType_FbxObject, &a0, &a1Keep, &a1))
        {
             ::FbxAnimCurveNode*sipRes;

            sipRes =  ::FbxAnimCurveNode::Create(a0,a1);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxAnimCurveNode,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_Create, doc_FbxAnimCurveNode_Create);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetClassId, "GetClassId(self) -> FbxClassId");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetClassId(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetClassId(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;
    bool sipSelfWasArg = (!sipSelf || sipIsDerivedClass((sipSimpleWrapper *)sipSelf));

    {
        const  ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp))
        {
             ::FbxClassId*sipRes;

            sipRes = new  ::FbxClassId((sipSelfWasArg ? sipCpp-> ::FbxAnimCurveNode::GetClassId() : sipCpp->GetClassId()));

            return sipConvertFromNewType(sipRes,sipType_FbxClassId,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetClassId, doc_FbxAnimCurveNode_GetClassId);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_IsAnimated, "IsAnimated(self, bool = False) -> bool");

extern "C" {static PyObject *meth_FbxAnimCurveNode_IsAnimated(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_IsAnimated(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0 = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B|b", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0))
        {
            bool sipRes;

            sipRes = sipCpp->IsAnimated(a0);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_IsAnimated, doc_FbxAnimCurveNode_IsAnimated);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetAnimationInterval, "GetAnimationInterval(self, FbxTimeSpan) -> bool");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetAnimationInterval(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetAnimationInterval(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxTimeSpan* a0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, sipType_FbxTimeSpan, &a0))
        {
            bool sipRes;

            sipRes = sipCpp->GetAnimationInterval(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetAnimationInterval, doc_FbxAnimCurveNode_GetAnimationInterval);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_IsComposite, "IsComposite(self) -> bool");

extern "C" {static PyObject *meth_FbxAnimCurveNode_IsComposite(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_IsComposite(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->IsComposite();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_IsComposite, doc_FbxAnimCurveNode_IsComposite);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_Find, "Find(self, str) -> FbxAnimCurveNode");

extern "C" {static PyObject *meth_FbxAnimCurveNode_Find(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_Find(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0))
        {
             ::FbxAnimCurveNode*sipRes;

            sipRes = sipCpp->Find(a0);
            Py_DECREF(a0Keep);

            return sipConvertFromType(sipRes,sipType_FbxAnimCurveNode,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_Find, doc_FbxAnimCurveNode_Find);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_CreateTypedCurveNode, "CreateTypedCurveNode(FbxProperty, FbxScene) -> FbxAnimCurveNode");

extern "C" {static PyObject *meth_FbxAnimCurveNode_CreateTypedCurveNode(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_CreateTypedCurveNode(PyObject *, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxProperty* a0;
         ::FbxScene* a1;

        if (sipParseArgs(&sipParseErr, sipArgs, "J9J8", sipType_FbxProperty, &a0, sipType_FbxScene, &a1))
        {
             ::FbxAnimCurveNode*sipRes;

            sipRes =  ::FbxAnimCurveNode::CreateTypedCurveNode(*a0,a1);

            return sipConvertFromType(sipRes,sipType_FbxAnimCurveNode,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_CreateTypedCurveNode, doc_FbxAnimCurveNode_CreateTypedCurveNode);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetChannelsCount, "GetChannelsCount(self) -> int");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetChannelsCount(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetChannelsCount(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp))
        {
            uint sipRes;

            sipRes = sipCpp->GetChannelsCount();

            return PyLong_FromUnsignedLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetChannelsCount, doc_FbxAnimCurveNode_GetChannelsCount);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetChannelIndex, "GetChannelIndex(self, str) -> int");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetChannelIndex(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetChannelIndex(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0))
        {
            int sipRes;

            sipRes = sipCpp->GetChannelIndex(a0);
            Py_DECREF(a0Keep);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetChannelIndex, doc_FbxAnimCurveNode_GetChannelIndex);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetChannelName, "GetChannelName(self, int) -> FbxString");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetChannelName(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetChannelName(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0))
        {
             ::FbxString*sipRes;

            sipRes = new  ::FbxString(sipCpp->GetChannelName(a0));

            return sipConvertFromNewType(sipRes,sipType_FbxString,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetChannelName, doc_FbxAnimCurveNode_GetChannelName);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_ResetChannels, "ResetChannels(self)");

extern "C" {static PyObject *meth_FbxAnimCurveNode_ResetChannels(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_ResetChannels(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp))
        {
            sipCpp->ResetChannels();

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_ResetChannels, doc_FbxAnimCurveNode_ResetChannels);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_AddChannel, "AddChannel(self, str, float) -> bool\n"
    "AddChannel(self, str, FbxString) -> bool");

extern "C" {static PyObject *meth_FbxAnimCurveNode_AddChannel(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_AddChannel(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
        double a1;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8d", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, &a1))
        {
            bool sipRes;

            sipRes = sipCpp->AddChannel(a0,a1);
            Py_DECREF(a0Keep);

            return PyBool_FromLong(sipRes);
        }
    }

    {
        const char* a0;
        PyObject *a0Keep;
         ::FbxString* a1;
        int a1State = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8J1", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, sipType_FbxString, &a1, &a1State))
        {
            bool sipRes;

            sipRes = sipCpp->AddChannel(a0,*a1);
            Py_DECREF(a0Keep);
            sipReleaseType(a1,sipType_FbxString,a1State);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_AddChannel, doc_FbxAnimCurveNode_AddChannel);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_SetChannelValue, "SetChannelValue(self, str, float)\n"
    "SetChannelValue(self, int, float)\n"
    "SetChannelValue(self, str, FbxString)\n"
    "SetChannelValue(self, int, FbxString)");

extern "C" {static PyObject *meth_FbxAnimCurveNode_SetChannelValue(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_SetChannelValue(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
        double a1;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8d", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, &a1))
        {
            sipCpp->SetChannelValue(a0,a1);
            Py_DECREF(a0Keep);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    {
        uint a0;
        double a1;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bud", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0, &a1))
        {
            sipCpp->SetChannelValue(a0,a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    {
        const char* a0;
        PyObject *a0Keep;
         ::FbxString* a1;
        int a1State = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8J1", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, sipType_FbxString, &a1, &a1State))
        {
            sipCpp->SetChannelValue(a0,*a1);
            Py_DECREF(a0Keep);
            sipReleaseType(a1,sipType_FbxString,a1State);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    {
        uint a0;
         ::FbxString* a1;
        int a1State = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BuJ1", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0, sipType_FbxString, &a1, &a1State))
        {
            sipCpp->SetChannelValue(a0,*a1);
            sipReleaseType(a1,sipType_FbxString,a1State);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_SetChannelValue, doc_FbxAnimCurveNode_SetChannelValue);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetChannelValue, "GetChannelValue(self, str, float) -> float\n"
    "GetChannelValue(self, int, float) -> float\n"
    "GetChannelValue(self, str, FbxString) -> FbxString\n"
    "GetChannelValue(self, int, FbxString) -> FbxString");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetChannelValue(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetChannelValue(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
        double a1;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8d", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, &a1))
        {
            double sipRes;

            sipRes = sipCpp->GetChannelValue(a0,a1);
            Py_DECREF(a0Keep);

            return PyFloat_FromDouble(sipRes);
        }
    }

    {
        uint a0;
        double a1;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bud", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0, &a1))
        {
            double sipRes;

            sipRes = sipCpp->GetChannelValue(a0,a1);

            return PyFloat_FromDouble(sipRes);
        }
    }

    {
        const char* a0;
        PyObject *a0Keep;
         ::FbxString* a1;
        int a1State = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8J1", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, sipType_FbxString, &a1, &a1State))
        {
             ::FbxString*sipRes;

            sipRes = new  ::FbxString(sipCpp->GetChannelValue(a0,*a1));
            Py_DECREF(a0Keep);
            sipReleaseType(a1,sipType_FbxString,a1State);

            return sipConvertFromNewType(sipRes,sipType_FbxString,NULL);
        }
    }

    {
        uint a0;
         ::FbxString* a1;
        int a1State = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BuJ1", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0, sipType_FbxString, &a1, &a1State))
        {
             ::FbxString*sipRes;

            sipRes = new  ::FbxString(sipCpp->GetChannelValue(a0,*a1));
            sipReleaseType(a1,sipType_FbxString,a1State);

            return sipConvertFromNewType(sipRes,sipType_FbxString,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetChannelValue, doc_FbxAnimCurveNode_GetChannelValue);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_DisconnectFromChannel, "DisconnectFromChannel(self, FbxAnimCurve, int) -> bool");

extern "C" {static PyObject *meth_FbxAnimCurveNode_DisconnectFromChannel(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_DisconnectFromChannel(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxAnimCurve* a0;
        uint a1;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8u", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, sipType_FbxAnimCurve, &a0, &a1))
        {
            bool sipRes;

            sipRes = sipCpp->DisconnectFromChannel(a0,a1);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_DisconnectFromChannel, doc_FbxAnimCurveNode_DisconnectFromChannel);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_ConnectToChannel, "ConnectToChannel(self, FbxAnimCurve, str, bool = False) -> bool\n"
    "ConnectToChannel(self, FbxAnimCurve, int, bool = False) -> bool");

extern "C" {static PyObject *meth_FbxAnimCurveNode_ConnectToChannel(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_ConnectToChannel(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxAnimCurve* a0;
        const char* a1;
        PyObject *a1Keep;
        bool a2 = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8A8|b", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, sipType_FbxAnimCurve, &a0, &a1Keep, &a1, &a2))
        {
            bool sipRes;

            sipRes = sipCpp->ConnectToChannel(a0,a1,a2);
            Py_DECREF(a1Keep);

            return PyBool_FromLong(sipRes);
        }
    }

    {
         ::FbxAnimCurve* a0;
        uint a1;
        bool a2 = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8u|b", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, sipType_FbxAnimCurve, &a0, &a1, &a2))
        {
            bool sipRes;

            sipRes = sipCpp->ConnectToChannel(a0,a1,a2);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_ConnectToChannel, doc_FbxAnimCurveNode_ConnectToChannel);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_CreateCurve, "CreateCurve(self, str, str) -> FbxAnimCurve\n"
    "CreateCurve(self, str, int = 0) -> FbxAnimCurve");

extern "C" {static PyObject *meth_FbxAnimCurveNode_CreateCurve(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_CreateCurve(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
        const char* a1;
        PyObject *a1Keep;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8A8", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, &a1Keep, &a1))
        {
             ::FbxAnimCurve*sipRes;

            sipRes = sipCpp->CreateCurve(a0,a1);
            Py_DECREF(a0Keep);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxAnimCurve,NULL);
        }
    }

    {
        const char* a0;
        PyObject *a0Keep;
        uint a1 = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8|u", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0Keep, &a0, &a1))
        {
             ::FbxAnimCurve*sipRes;

            sipRes = sipCpp->CreateCurve(a0,a1);
            Py_DECREF(a0Keep);

            return sipConvertFromType(sipRes,sipType_FbxAnimCurve,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_CreateCurve, doc_FbxAnimCurveNode_CreateCurve);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetCurveCount, "GetCurveCount(self, int, str = None) -> int");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetCurveCount(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetCurveCount(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        uint a0;
        const char* a1 = 0;
        PyObject *a1Keep = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bu|A8", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0, &a1Keep, &a1))
        {
            int sipRes;

            sipRes = sipCpp->GetCurveCount(a0,a1);
            Py_XDECREF(a1Keep);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetCurveCount, doc_FbxAnimCurveNode_GetCurveCount);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAnimCurveNode_GetCurve, "GetCurve(self, int, int = 0, str = None) -> FbxAnimCurve");

extern "C" {static PyObject *meth_FbxAnimCurveNode_GetCurve(PyObject *, PyObject *);}
static PyObject *meth_FbxAnimCurveNode_GetCurve(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        uint a0;
        uint a1 = 0;
        const char* a2 = 0;
        PyObject *a2Keep = 0;
         ::FbxAnimCurveNode *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bu|uA8", &sipSelf, sipType_FbxAnimCurveNode, &sipCpp, &a0, &a1, &a2Keep, &a2))
        {
             ::FbxAnimCurve*sipRes;

            sipRes = sipCpp->GetCurve(a0,a1,a2);
            Py_XDECREF(a2Keep);

            return sipConvertFromType(sipRes,sipType_FbxAnimCurve,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAnimCurveNode, sipName_GetCurve, doc_FbxAnimCurveNode_GetCurve);

    return NULL;
}


extern "C" {static PyObject *slot_FbxAnimCurveNode___ne__(PyObject *,PyObject *);}
static PyObject *slot_FbxAnimCurveNode___ne__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxAnimCurveNode *sipCpp = reinterpret_cast< ::FbxAnimCurveNode *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxAnimCurveNode));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxAnimCurveNode* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxAnimCurveNode, &a0))
        {
            bool sipRes;

            sipRes = !sipCpp-> ::FbxAnimCurveNode::operator==(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, ne_slot, sipType_FbxAnimCurveNode, sipSelf, sipArg);
}


extern "C" {static PyObject *slot_FbxAnimCurveNode___eq__(PyObject *,PyObject *);}
static PyObject *slot_FbxAnimCurveNode___eq__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxAnimCurveNode *sipCpp = reinterpret_cast< ::FbxAnimCurveNode *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxAnimCurveNode));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxAnimCurveNode* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxAnimCurveNode, &a0))
        {
            bool sipRes;

            sipRes = sipCpp-> ::FbxAnimCurveNode::operator==(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, eq_slot, sipType_FbxAnimCurveNode, sipSelf, sipArg);
}


/* Cast a pointer to a type somewhere in its inheritance hierarchy. */
extern "C" {static void *cast_FbxAnimCurveNode(void *, const sipTypeDef *);}
static void *cast_FbxAnimCurveNode(void *sipCppV, const sipTypeDef *targetType)
{
     ::FbxAnimCurveNode *sipCpp = reinterpret_cast< ::FbxAnimCurveNode *>(sipCppV);

    if (targetType == sipType_FbxObject)
        return static_cast< ::FbxObject *>(sipCpp);

    return sipCppV;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxAnimCurveNode(void *, int);}
static void release_FbxAnimCurveNode(void *, int)
{
}


/* Define this type's super-types. */
static sipEncodedTypeDef supers_FbxAnimCurveNode[] = {{244, 255, 1}};


/* Define this type's Python slots. */
static sipPySlotDef slots_FbxAnimCurveNode[] = {
    {(void *)slot_FbxAnimCurveNode___ne__, ne_slot},
    {(void *)slot_FbxAnimCurveNode___eq__, eq_slot},
    {0, (sipPySlotType)0}
};


static PyMethodDef methods_FbxAnimCurveNode[] = {
    {SIP_MLNAME_CAST(sipName_AddChannel), meth_FbxAnimCurveNode_AddChannel, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_AddChannel)},
    {SIP_MLNAME_CAST(sipName_ConnectToChannel), meth_FbxAnimCurveNode_ConnectToChannel, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_ConnectToChannel)},
    {SIP_MLNAME_CAST(sipName_Create), meth_FbxAnimCurveNode_Create, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_Create)},
    {SIP_MLNAME_CAST(sipName_CreateCurve), meth_FbxAnimCurveNode_CreateCurve, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_CreateCurve)},
    {SIP_MLNAME_CAST(sipName_CreateTypedCurveNode), meth_FbxAnimCurveNode_CreateTypedCurveNode, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_CreateTypedCurveNode)},
    {SIP_MLNAME_CAST(sipName_DisconnectFromChannel), meth_FbxAnimCurveNode_DisconnectFromChannel, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_DisconnectFromChannel)},
    {SIP_MLNAME_CAST(sipName_Find), meth_FbxAnimCurveNode_Find, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_Find)},
    {SIP_MLNAME_CAST(sipName_GetAnimationInterval), meth_FbxAnimCurveNode_GetAnimationInterval, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetAnimationInterval)},
    {SIP_MLNAME_CAST(sipName_GetChannelIndex), meth_FbxAnimCurveNode_GetChannelIndex, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetChannelIndex)},
    {SIP_MLNAME_CAST(sipName_GetChannelName), meth_FbxAnimCurveNode_GetChannelName, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetChannelName)},
    {SIP_MLNAME_CAST(sipName_GetChannelValue), meth_FbxAnimCurveNode_GetChannelValue, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetChannelValue)},
    {SIP_MLNAME_CAST(sipName_GetChannelsCount), meth_FbxAnimCurveNode_GetChannelsCount, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetChannelsCount)},
    {SIP_MLNAME_CAST(sipName_GetClassId), meth_FbxAnimCurveNode_GetClassId, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetClassId)},
    {SIP_MLNAME_CAST(sipName_GetCurve), meth_FbxAnimCurveNode_GetCurve, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetCurve)},
    {SIP_MLNAME_CAST(sipName_GetCurveCount), meth_FbxAnimCurveNode_GetCurveCount, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_GetCurveCount)},
    {SIP_MLNAME_CAST(sipName_IsAnimated), meth_FbxAnimCurveNode_IsAnimated, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_IsAnimated)},
    {SIP_MLNAME_CAST(sipName_IsComposite), meth_FbxAnimCurveNode_IsComposite, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_IsComposite)},
    {SIP_MLNAME_CAST(sipName_ResetChannels), meth_FbxAnimCurveNode_ResetChannels, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_ResetChannels)},
    {SIP_MLNAME_CAST(sipName_SetChannelValue), meth_FbxAnimCurveNode_SetChannelValue, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAnimCurveNode_SetChannelValue)}
};


extern "C" {static PyObject *varget_FbxAnimCurveNode_ClassId(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxAnimCurveNode_ClassId(void *, PyObject *, PyObject *)
{
     ::FbxClassId*sipVal;

    sipVal = & ::FbxAnimCurveNode::ClassId;

    return sipConvertFromType(sipVal, sipType_FbxClassId, NULL);
}


extern "C" {static int varset_FbxAnimCurveNode_ClassId(void *, PyObject *, PyObject *);}
static int varset_FbxAnimCurveNode_ClassId(void *, PyObject *sipPy, PyObject *)
{
     ::FbxClassId*sipVal;
    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxClassId *>(sipForceConvertToType(sipPy,sipType_FbxClassId,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

     ::FbxAnimCurveNode::ClassId = *sipVal;

    return 0;
}

sipVariableDef variables_FbxAnimCurveNode[] = {
    {ClassVariable, sipName_ClassId, (PyMethodDef *)varget_FbxAnimCurveNode_ClassId, (PyMethodDef *)varset_FbxAnimCurveNode_ClassId, NULL, NULL},
};


sipClassTypeDef sipTypeDef_fbx_FbxAnimCurveNode = {
    {
        -1,
        0,
        0,
        SIP_TYPE_SCC|SIP_TYPE_CLASS,
        sipNameNr_FbxAnimCurveNode,
        {0},
        0
    },
    {
        sipNameNr_FbxAnimCurveNode,
        {0, 0, 1},
        19, methods_FbxAnimCurveNode,
        0, 0,
        1, variables_FbxAnimCurveNode,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    0,
    -1,
    -1,
    supers_FbxAnimCurveNode,
    slots_FbxAnimCurveNode,
    0,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    0,
    0,
    0,
    0,
    release_FbxAnimCurveNode,
    cast_FbxAnimCurveNode,
    0,
    0,
    0,
    0,
    0,
    0
};
