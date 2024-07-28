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




PyDoc_STRVAR(doc_FbxAudio_Create, "Create(FbxManager, str) -> FbxAudio\n"
    "Create(FbxObject, str) -> FbxAudio");

extern "C" {static PyObject *meth_FbxAudio_Create(PyObject *, PyObject *);}
static PyObject *meth_FbxAudio_Create(PyObject *, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxManager* a0;
        const char* a1;
        PyObject *a1Keep;

        if (sipParseArgs(&sipParseErr, sipArgs, "J8A8", sipType_FbxManager, &a0, &a1Keep, &a1))
        {
             ::FbxAudio*sipRes;

            sipRes =  ::FbxAudio::Create(a0,a1);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxAudio,NULL);
        }
    }

    {
         ::FbxObject* a0;
        const char* a1;
        PyObject *a1Keep;

        if (sipParseArgs(&sipParseErr, sipArgs, "J8A8", sipType_FbxObject, &a0, &a1Keep, &a1))
        {
             ::FbxAudio*sipRes;

            sipRes =  ::FbxAudio::Create(a0,a1);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxAudio,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAudio, sipName_Create, doc_FbxAudio_Create);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAudio_GetClassId, "GetClassId(self) -> FbxClassId");

extern "C" {static PyObject *meth_FbxAudio_GetClassId(PyObject *, PyObject *);}
static PyObject *meth_FbxAudio_GetClassId(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;
    bool sipSelfWasArg = (!sipSelf || sipIsDerivedClass((sipSimpleWrapper *)sipSelf));

    {
        const  ::FbxAudio *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxAudio, &sipCpp))
        {
             ::FbxClassId*sipRes;

            sipRes = new  ::FbxClassId((sipSelfWasArg ? sipCpp-> ::FbxAudio::GetClassId() : sipCpp->GetClassId()));

            return sipConvertFromNewType(sipRes,sipType_FbxClassId,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAudio, sipName_GetClassId, doc_FbxAudio_GetClassId);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAudio_Reset, "Reset(self)");

extern "C" {static PyObject *meth_FbxAudio_Reset(PyObject *, PyObject *);}
static PyObject *meth_FbxAudio_Reset(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxAudio *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxAudio, &sipCpp))
        {
            sipCpp->Reset();

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAudio, sipName_Reset, doc_FbxAudio_Reset);

    return NULL;
}


PyDoc_STRVAR(doc_FbxAudio_Volume, "Volume(self) -> FbxProperty");

extern "C" {static PyObject *meth_FbxAudio_Volume(PyObject *, PyObject *);}
static PyObject *meth_FbxAudio_Volume(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxAudio *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxAudio, &sipCpp))
        {
             ::FbxProperty*sipRes;

            sipRes = new  ::FbxProperty(sipCpp->Volume());

            return sipConvertFromNewType(sipRes,sipType_FbxProperty,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxAudio, sipName_Volume, doc_FbxAudio_Volume);

    return NULL;
}


extern "C" {static PyObject *slot_FbxAudio___ne__(PyObject *,PyObject *);}
static PyObject *slot_FbxAudio___ne__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxAudio));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxAudio* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxAudio, &a0))
        {
            bool sipRes;

            sipRes = !sipCpp-> ::FbxAudio::operator==(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, ne_slot, sipType_FbxAudio, sipSelf, sipArg);
}


extern "C" {static PyObject *slot_FbxAudio___eq__(PyObject *,PyObject *);}
static PyObject *slot_FbxAudio___eq__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxAudio));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxAudio* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxAudio, &a0))
        {
            bool sipRes;

            sipRes = sipCpp-> ::FbxAudio::operator==(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, eq_slot, sipType_FbxAudio, sipSelf, sipArg);
}


/* Cast a pointer to a type somewhere in its inheritance hierarchy. */
extern "C" {static void *cast_FbxAudio(void *, const sipTypeDef *);}
static void *cast_FbxAudio(void *sipCppV, const sipTypeDef *targetType)
{
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipCppV);

    if (targetType == sipType_FbxMediaClip)
        return static_cast< ::FbxMediaClip *>(sipCpp);

    if (targetType == sipType_FbxObject)
        return static_cast< ::FbxObject *>(sipCpp);

    return sipCppV;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxAudio(void *, int);}
static void release_FbxAudio(void *, int)
{
}


/* Define this type's super-types. */
static sipEncodedTypeDef supers_FbxAudio[] = {{221, 255, 1}};


/* Define this type's Python slots. */
static sipPySlotDef slots_FbxAudio[] = {
    {(void *)slot_FbxAudio___ne__, ne_slot},
    {(void *)slot_FbxAudio___eq__, eq_slot},
    {0, (sipPySlotType)0}
};


static PyMethodDef methods_FbxAudio[] = {
    {SIP_MLNAME_CAST(sipName_Create), meth_FbxAudio_Create, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAudio_Create)},
    {SIP_MLNAME_CAST(sipName_GetClassId), meth_FbxAudio_GetClassId, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAudio_GetClassId)},
    {SIP_MLNAME_CAST(sipName_Reset), meth_FbxAudio_Reset, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAudio_Reset)},
    {SIP_MLNAME_CAST(sipName_Volume), meth_FbxAudio_Volume, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxAudio_Volume)}
};


extern "C" {static PyObject *varget_FbxAudio_AnimFX(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxAudio_AnimFX(void *sipSelf, PyObject *, PyObject *)
{
     ::FbxProperty*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    sipVal = &sipCpp->AnimFX;

    return sipConvertFromType(sipVal, sipType_FbxProperty, NULL);
}


extern "C" {static int varset_FbxAudio_AnimFX(void *, PyObject *, PyObject *);}
static int varset_FbxAudio_AnimFX(void *sipSelf, PyObject *sipPy, PyObject *)
{
     ::FbxProperty*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxProperty *>(sipForceConvertToType(sipPy,sipType_FbxProperty,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

    sipCpp->AnimFX = *sipVal;

    return 0;
}


extern "C" {static PyObject *varget_FbxAudio_BitRate(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxAudio_BitRate(void *sipSelf, PyObject *, PyObject *)
{
     ::FbxPropertyInteger1*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    sipVal = &sipCpp->BitRate;

    return sipConvertFromType(sipVal, sipType_FbxPropertyInteger1, NULL);
}


extern "C" {static int varset_FbxAudio_BitRate(void *, PyObject *, PyObject *);}
static int varset_FbxAudio_BitRate(void *sipSelf, PyObject *sipPy, PyObject *)
{
     ::FbxPropertyInteger1*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxPropertyInteger1 *>(sipForceConvertToType(sipPy,sipType_FbxPropertyInteger1,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

    sipCpp->BitRate = *sipVal;

    return 0;
}


extern "C" {static PyObject *varget_FbxAudio_Channels(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxAudio_Channels(void *sipSelf, PyObject *, PyObject *)
{
     ::FbxPropertyUChar1*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    sipVal = &sipCpp->Channels;

    return sipConvertFromType(sipVal, sipType_FbxPropertyUChar1, NULL);
}


extern "C" {static int varset_FbxAudio_Channels(void *, PyObject *, PyObject *);}
static int varset_FbxAudio_Channels(void *sipSelf, PyObject *sipPy, PyObject *)
{
     ::FbxPropertyUChar1*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxPropertyUChar1 *>(sipForceConvertToType(sipPy,sipType_FbxPropertyUChar1,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

    sipCpp->Channels = *sipVal;

    return 0;
}


extern "C" {static PyObject *varget_FbxAudio_ClassId(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxAudio_ClassId(void *, PyObject *, PyObject *)
{
     ::FbxClassId*sipVal;

    sipVal = & ::FbxAudio::ClassId;

    return sipConvertFromType(sipVal, sipType_FbxClassId, NULL);
}


extern "C" {static int varset_FbxAudio_ClassId(void *, PyObject *, PyObject *);}
static int varset_FbxAudio_ClassId(void *, PyObject *sipPy, PyObject *)
{
     ::FbxClassId*sipVal;
    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxClassId *>(sipForceConvertToType(sipPy,sipType_FbxClassId,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

     ::FbxAudio::ClassId = *sipVal;

    return 0;
}


extern "C" {static PyObject *varget_FbxAudio_Duration(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxAudio_Duration(void *sipSelf, PyObject *, PyObject *)
{
     ::FbxPropertyFbxTime*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    sipVal = &sipCpp->Duration;

    return sipConvertFromType(sipVal, sipType_FbxPropertyFbxTime, NULL);
}


extern "C" {static int varset_FbxAudio_Duration(void *, PyObject *, PyObject *);}
static int varset_FbxAudio_Duration(void *sipSelf, PyObject *sipPy, PyObject *)
{
     ::FbxPropertyFbxTime*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxPropertyFbxTime *>(sipForceConvertToType(sipPy,sipType_FbxPropertyFbxTime,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

    sipCpp->Duration = *sipVal;

    return 0;
}


extern "C" {static PyObject *varget_FbxAudio_SampleRate(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxAudio_SampleRate(void *sipSelf, PyObject *, PyObject *)
{
     ::FbxPropertyInteger1*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    sipVal = &sipCpp->SampleRate;

    return sipConvertFromType(sipVal, sipType_FbxPropertyInteger1, NULL);
}


extern "C" {static int varset_FbxAudio_SampleRate(void *, PyObject *, PyObject *);}
static int varset_FbxAudio_SampleRate(void *sipSelf, PyObject *sipPy, PyObject *)
{
     ::FbxPropertyInteger1*sipVal;
     ::FbxAudio *sipCpp = reinterpret_cast< ::FbxAudio *>(sipSelf);

    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxPropertyInteger1 *>(sipForceConvertToType(sipPy,sipType_FbxPropertyInteger1,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

    sipCpp->SampleRate = *sipVal;

    return 0;
}

sipVariableDef variables_FbxAudio[] = {
    {InstanceVariable, sipName_AnimFX, (PyMethodDef *)varget_FbxAudio_AnimFX, (PyMethodDef *)varset_FbxAudio_AnimFX, NULL, NULL},
    {InstanceVariable, sipName_BitRate, (PyMethodDef *)varget_FbxAudio_BitRate, (PyMethodDef *)varset_FbxAudio_BitRate, NULL, NULL},
    {InstanceVariable, sipName_Channels, (PyMethodDef *)varget_FbxAudio_Channels, (PyMethodDef *)varset_FbxAudio_Channels, NULL, NULL},
    {ClassVariable, sipName_ClassId, (PyMethodDef *)varget_FbxAudio_ClassId, (PyMethodDef *)varset_FbxAudio_ClassId, NULL, NULL},
    {InstanceVariable, sipName_Duration, (PyMethodDef *)varget_FbxAudio_Duration, (PyMethodDef *)varset_FbxAudio_Duration, NULL, NULL},
    {InstanceVariable, sipName_SampleRate, (PyMethodDef *)varget_FbxAudio_SampleRate, (PyMethodDef *)varset_FbxAudio_SampleRate, NULL, NULL},
};


sipClassTypeDef sipTypeDef_fbx_FbxAudio = {
    {
        -1,
        0,
        0,
        SIP_TYPE_SCC|SIP_TYPE_CLASS,
        sipNameNr_FbxAudio,
        {0},
        0
    },
    {
        sipNameNr_FbxAudio,
        {0, 0, 1},
        4, methods_FbxAudio,
        0, 0,
        6, variables_FbxAudio,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    0,
    -1,
    -1,
    supers_FbxAudio,
    slots_FbxAudio,
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
    release_FbxAudio,
    cast_FbxAudio,
    0,
    0,
    0,
    0,
    0,
    0
};
