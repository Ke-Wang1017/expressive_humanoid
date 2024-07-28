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


#line 4 "/home/ke/Documents/expressive-humanoid/sip/fbxstatus.sip"
#include <fbxsdk.h>
#line 44 "/home/ke/Documents/expressive-humanoid/build/Python38_x64/sipfbxFbxLine.cpp"


PyDoc_STRVAR(doc_FbxLine_Create, "Create(FbxManager, str) -> FbxLine\n"
    "Create(FbxObject, str) -> FbxLine");

extern "C" {static PyObject *meth_FbxLine_Create(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_Create(PyObject *, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxManager* a0;
        const char* a1;
        PyObject *a1Keep;

        if (sipParseArgs(&sipParseErr, sipArgs, "J8A8", sipType_FbxManager, &a0, &a1Keep, &a1))
        {
             ::FbxLine*sipRes;

            sipRes =  ::FbxLine::Create(a0,a1);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxLine,NULL);
        }
    }

    {
         ::FbxObject* a0;
        const char* a1;
        PyObject *a1Keep;

        if (sipParseArgs(&sipParseErr, sipArgs, "J8A8", sipType_FbxObject, &a0, &a1Keep, &a1))
        {
             ::FbxLine*sipRes;

            sipRes =  ::FbxLine::Create(a0,a1);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxLine,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_Create, doc_FbxLine_Create);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetClassId, "GetClassId(self) -> FbxClassId");

extern "C" {static PyObject *meth_FbxLine_GetClassId(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetClassId(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;
    bool sipSelfWasArg = (!sipSelf || sipIsDerivedClass((sipSimpleWrapper *)sipSelf));

    {
        const  ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLine, &sipCpp))
        {
             ::FbxClassId*sipRes;

            sipRes = new  ::FbxClassId((sipSelfWasArg ? sipCpp-> ::FbxLine::GetClassId() : sipCpp->GetClassId()));

            return sipConvertFromNewType(sipRes,sipType_FbxClassId,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetClassId, doc_FbxLine_GetClassId);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetAttributeType, "GetAttributeType(self) -> FbxNodeAttribute.EType");

extern "C" {static PyObject *meth_FbxLine_GetAttributeType(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetAttributeType(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;
    bool sipSelfWasArg = (!sipSelf || sipIsDerivedClass((sipSimpleWrapper *)sipSelf));

    {
        const  ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLine, &sipCpp))
        {
             ::FbxNodeAttribute::EType sipRes;

            sipRes = (sipSelfWasArg ? sipCpp-> ::FbxLine::GetAttributeType() : sipCpp->GetAttributeType());

            return sipConvertFromEnum(sipRes,sipType_FbxNodeAttribute_EType);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetAttributeType, doc_FbxLine_GetAttributeType);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_Reset, "Reset(self)");

extern "C" {static PyObject *meth_FbxLine_Reset(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_Reset(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLine, &sipCpp))
        {
            sipCpp->Reset();

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_Reset, doc_FbxLine_Reset);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_SetIndexArraySize, "SetIndexArraySize(self, int)");

extern "C" {static PyObject *meth_FbxLine_SetIndexArraySize(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_SetIndexArraySize(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLine, &sipCpp, &a0))
        {
            sipCpp->SetIndexArraySize(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_SetIndexArraySize, doc_FbxLine_SetIndexArraySize);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetIndexArraySize, "GetIndexArraySize(self) -> int");

extern "C" {static PyObject *meth_FbxLine_GetIndexArraySize(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetIndexArraySize(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLine, &sipCpp))
        {
            int sipRes;

            sipRes = sipCpp->GetIndexArraySize();

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetIndexArraySize, doc_FbxLine_GetIndexArraySize);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetIndexArray, "GetIndexArray(self) -> IntArray");

extern "C" {static PyObject *meth_FbxLine_GetIndexArray(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetIndexArray(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLine, &sipCpp))
        {
             ::IntArray*sipRes;

            sipRes = sipCpp->GetIndexArray();

            return sipConvertFromType(sipRes,sipType_IntArray,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetIndexArray, doc_FbxLine_GetIndexArray);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_SetPointIndexAt, "SetPointIndexAt(self, int, int, bool = False) -> bool");

extern "C" {static PyObject *meth_FbxLine_SetPointIndexAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_SetPointIndexAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        int a1;
        bool a2 = 0;
         ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bii|b", &sipSelf, sipType_FbxLine, &sipCpp, &a0, &a1, &a2))
        {
            bool sipRes;

            sipRes = sipCpp->SetPointIndexAt(a0,a1,a2);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_SetPointIndexAt, doc_FbxLine_SetPointIndexAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetPointIndexAt, "GetPointIndexAt(self, int) -> int");

extern "C" {static PyObject *meth_FbxLine_GetPointIndexAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetPointIndexAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        const  ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLine, &sipCpp, &a0))
        {
            int sipRes;

            sipRes = sipCpp->GetPointIndexAt(a0);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetPointIndexAt, doc_FbxLine_GetPointIndexAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_AddPointIndex, "AddPointIndex(self, int, bool = False) -> bool");

extern "C" {static PyObject *meth_FbxLine_AddPointIndex(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_AddPointIndex(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        bool a1 = 0;
         ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi|b", &sipSelf, sipType_FbxLine, &sipCpp, &a0, &a1))
        {
            bool sipRes;

            sipRes = sipCpp->AddPointIndex(a0,a1);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_AddPointIndex, doc_FbxLine_AddPointIndex);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetEndPointArray, "GetEndPointArray(self) -> IntArray");

extern "C" {static PyObject *meth_FbxLine_GetEndPointArray(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetEndPointArray(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLine, &sipCpp))
        {
             ::IntArray*sipRes;

            sipRes = sipCpp->GetEndPointArray();

            return sipConvertFromType(sipRes,sipType_IntArray,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetEndPointArray, doc_FbxLine_GetEndPointArray);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_AddEndPoint, "AddEndPoint(self, int) -> bool");

extern "C" {static PyObject *meth_FbxLine_AddEndPoint(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_AddEndPoint(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLine, &sipCpp, &a0))
        {
            bool sipRes;

            sipRes = sipCpp->AddEndPoint(a0);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_AddEndPoint, doc_FbxLine_AddEndPoint);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetEndPointAt, "GetEndPointAt(self, int) -> int");

extern "C" {static PyObject *meth_FbxLine_GetEndPointAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetEndPointAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        const  ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLine, &sipCpp, &a0))
        {
            int sipRes;

            sipRes = sipCpp->GetEndPointAt(a0);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetEndPointAt, doc_FbxLine_GetEndPointAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLine_GetEndPointCount, "GetEndPointCount(self) -> int");

extern "C" {static PyObject *meth_FbxLine_GetEndPointCount(PyObject *, PyObject *);}
static PyObject *meth_FbxLine_GetEndPointCount(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLine *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLine, &sipCpp))
        {
            int sipRes;

            sipRes = sipCpp->GetEndPointCount();

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLine, sipName_GetEndPointCount, doc_FbxLine_GetEndPointCount);

    return NULL;
}


extern "C" {static PyObject *slot_FbxLine___ne__(PyObject *,PyObject *);}
static PyObject *slot_FbxLine___ne__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxLine *sipCpp = reinterpret_cast< ::FbxLine *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxLine));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLine* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxLine, &a0))
        {
            bool sipRes;

            sipRes = !sipCpp-> ::FbxLine::operator==(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, ne_slot, sipType_FbxLine, sipSelf, sipArg);
}


extern "C" {static PyObject *slot_FbxLine___eq__(PyObject *,PyObject *);}
static PyObject *slot_FbxLine___eq__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxLine *sipCpp = reinterpret_cast< ::FbxLine *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxLine));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLine* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxLine, &a0))
        {
            bool sipRes;

            sipRes = sipCpp-> ::FbxLine::operator==(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, eq_slot, sipType_FbxLine, sipSelf, sipArg);
}


/* Cast a pointer to a type somewhere in its inheritance hierarchy. */
extern "C" {static void *cast_FbxLine(void *, const sipTypeDef *);}
static void *cast_FbxLine(void *sipCppV, const sipTypeDef *targetType)
{
     ::FbxLine *sipCpp = reinterpret_cast< ::FbxLine *>(sipCppV);

    if (targetType == sipType_FbxGeometry)
        return static_cast< ::FbxGeometry *>(sipCpp);

    if (targetType == sipType_FbxGeometryBase)
        return static_cast< ::FbxGeometryBase *>(sipCpp);

    if (targetType == sipType_FbxLayerContainer)
        return static_cast< ::FbxLayerContainer *>(sipCpp);

    if (targetType == sipType_FbxNodeAttribute)
        return static_cast< ::FbxNodeAttribute *>(sipCpp);

    if (targetType == sipType_FbxObject)
        return static_cast< ::FbxObject *>(sipCpp);

    return sipCppV;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxLine(void *, int);}
static void release_FbxLine(void *, int)
{
}


/* Define this type's super-types. */
static sipEncodedTypeDef supers_FbxLine[] = {{147, 255, 1}};


/* Define this type's Python slots. */
static sipPySlotDef slots_FbxLine[] = {
    {(void *)slot_FbxLine___ne__, ne_slot},
    {(void *)slot_FbxLine___eq__, eq_slot},
    {0, (sipPySlotType)0}
};


static PyMethodDef methods_FbxLine[] = {
    {SIP_MLNAME_CAST(sipName_AddEndPoint), meth_FbxLine_AddEndPoint, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_AddEndPoint)},
    {SIP_MLNAME_CAST(sipName_AddPointIndex), meth_FbxLine_AddPointIndex, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_AddPointIndex)},
    {SIP_MLNAME_CAST(sipName_Create), meth_FbxLine_Create, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_Create)},
    {SIP_MLNAME_CAST(sipName_GetAttributeType), meth_FbxLine_GetAttributeType, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetAttributeType)},
    {SIP_MLNAME_CAST(sipName_GetClassId), meth_FbxLine_GetClassId, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetClassId)},
    {SIP_MLNAME_CAST(sipName_GetEndPointArray), meth_FbxLine_GetEndPointArray, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetEndPointArray)},
    {SIP_MLNAME_CAST(sipName_GetEndPointAt), meth_FbxLine_GetEndPointAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetEndPointAt)},
    {SIP_MLNAME_CAST(sipName_GetEndPointCount), meth_FbxLine_GetEndPointCount, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetEndPointCount)},
    {SIP_MLNAME_CAST(sipName_GetIndexArray), meth_FbxLine_GetIndexArray, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetIndexArray)},
    {SIP_MLNAME_CAST(sipName_GetIndexArraySize), meth_FbxLine_GetIndexArraySize, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetIndexArraySize)},
    {SIP_MLNAME_CAST(sipName_GetPointIndexAt), meth_FbxLine_GetPointIndexAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_GetPointIndexAt)},
    {SIP_MLNAME_CAST(sipName_Reset), meth_FbxLine_Reset, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_Reset)},
    {SIP_MLNAME_CAST(sipName_SetIndexArraySize), meth_FbxLine_SetIndexArraySize, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_SetIndexArraySize)},
    {SIP_MLNAME_CAST(sipName_SetPointIndexAt), meth_FbxLine_SetPointIndexAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLine_SetPointIndexAt)}
};


extern "C" {static PyObject *varget_FbxLine_ClassId(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxLine_ClassId(void *, PyObject *, PyObject *)
{
     ::FbxClassId*sipVal;

    sipVal = & ::FbxLine::ClassId;

    return sipConvertFromType(sipVal, sipType_FbxClassId, NULL);
}


extern "C" {static int varset_FbxLine_ClassId(void *, PyObject *, PyObject *);}
static int varset_FbxLine_ClassId(void *, PyObject *sipPy, PyObject *)
{
     ::FbxClassId*sipVal;
    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxClassId *>(sipForceConvertToType(sipPy,sipType_FbxClassId,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

     ::FbxLine::ClassId = *sipVal;

    return 0;
}


extern "C" {static PyObject *varget_FbxLine_Renderable(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxLine_Renderable(void *sipSelf, PyObject *, PyObject *)
{
     ::FbxPropertyBool1*sipVal;
     ::FbxLine *sipCpp = reinterpret_cast< ::FbxLine *>(sipSelf);

    sipVal = &sipCpp->Renderable;

    return sipConvertFromType(sipVal, sipType_FbxPropertyBool1, NULL);
}


extern "C" {static int varset_FbxLine_Renderable(void *, PyObject *, PyObject *);}
static int varset_FbxLine_Renderable(void *sipSelf, PyObject *sipPy, PyObject *)
{
     ::FbxPropertyBool1*sipVal;
     ::FbxLine *sipCpp = reinterpret_cast< ::FbxLine *>(sipSelf);

    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxPropertyBool1 *>(sipForceConvertToType(sipPy,sipType_FbxPropertyBool1,NULL,SIP_NOT_NONE,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

    sipCpp->Renderable = *sipVal;

    return 0;
}

sipVariableDef variables_FbxLine[] = {
    {ClassVariable, sipName_ClassId, (PyMethodDef *)varget_FbxLine_ClassId, (PyMethodDef *)varset_FbxLine_ClassId, NULL, NULL},
    {InstanceVariable, sipName_Renderable, (PyMethodDef *)varget_FbxLine_Renderable, (PyMethodDef *)varset_FbxLine_Renderable, NULL, NULL},
};


sipClassTypeDef sipTypeDef_fbx_FbxLine = {
    {
        -1,
        0,
        0,
        SIP_TYPE_SCC|SIP_TYPE_CLASS,
        sipNameNr_FbxLine,
        {0},
        0
    },
    {
        sipNameNr_FbxLine,
        {0, 0, 1},
        14, methods_FbxLine,
        0, 0,
        2, variables_FbxLine,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    0,
    -1,
    -1,
    supers_FbxLine,
    slots_FbxLine,
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
    release_FbxLine,
    cast_FbxLine,
    0,
    0,
    0,
    0,
    0,
    0
};
